"""
Training pipeline for FinBERT hierarchical classifier.

All configuration lives in finbert/config.toml — no CLI arguments.

Usage:
    python -m finbert.train
"""

from __future__ import annotations

import json
import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from finbert.config import FinBERTConfig, load_config
from finbert.dataset import NewsClassificationDataset
from finbert.hierarchy import build_label_maps
from finbert.model import load_finbert_classifier
from finbert.wandb_handler import WandbHandler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run evaluation, return loss / accuracy metrics."""
    model.eval()
    total_loss = 0.0
    l1_correct = 0
    l2_correct = 0
    sent_correct = 0
    total = 0
    all_l1_true: list[int] = []
    all_l1_pred: list[int] = []
    all_l2_true: list[int] = []
    all_l2_pred: list[int] = []
    all_sent_true: list[int] = []
    all_sent_pred: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                l1_label=batch["l1_label"],
                l2_label=batch["l2_label"],
                sentiment_label=batch["sentiment_label"],
            )
            total_loss += outputs["loss"].item() * batch["input_ids"].size(0)
            l1_preds = outputs["l1_logits"].argmax(dim=-1)
            l2_preds = outputs["l2_logits"].argmax(dim=-1)
            sent_preds = outputs["sentiment_logits"].argmax(dim=-1)
            l1_correct += (l1_preds == batch["l1_label"]).sum().item()
            l2_correct += (l2_preds == batch["l2_label"]).sum().item()
            sent_correct += (sent_preds == batch["sentiment_label"]).sum().item()
            total += batch["input_ids"].size(0)
            all_l1_true.extend(batch["l1_label"].cpu().tolist())
            all_l1_pred.extend(l1_preds.cpu().tolist())
            all_l2_true.extend(batch["l2_label"].cpu().tolist())
            all_l2_pred.extend(l2_preds.cpu().tolist())
            all_sent_true.extend(batch["sentiment_label"].cpu().tolist())
            all_sent_pred.extend(sent_preds.cpu().tolist())

    return {
        "loss": total_loss / total,
        "l1_accuracy": l1_correct / total,
        "l2_accuracy": l2_correct / total,
        "sentiment_accuracy": sent_correct / total,
        "_l1_true": all_l1_true,
        "_l1_pred": all_l1_pred,
        "_l2_true": all_l2_true,
        "_l2_pred": all_l2_pred,
        "_sent_true": all_sent_true,
        "_sent_pred": all_sent_pred,
    }


def train(cfg: FinBERTConfig) -> None:
    mcfg = cfg.model
    tcfg = cfg.training
    dcfg = cfg.data
    ocfg = cfg.output
    hcfg = cfg.hierarchy
    lcfg = cfg.loss

    set_seed(tcfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── W&B ────────────────────────────────────────────────────────────
    wb = WandbHandler(cfg)
    wb.init_run()

    # ── Label maps ─────────────────────────────────────
    l1_to_idx, idx_to_l1, l2_to_idx, idx_to_l2, _ = build_label_maps()
    assert len(l1_to_idx) == hcfg.num_level1, f"Hierarchy has {len(l1_to_idx)} L1, config expects {hcfg.num_level1}"
    assert len(l2_to_idx) == hcfg.num_level2, f"Hierarchy has {len(l2_to_idx)} L2, config expects {hcfg.num_level2}"

    # ── Tokenizer ──────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(mcfg.pretrained_model)

    # ── Datasets ───────────────────────────────────────
    train_ds = NewsClassificationDataset(
        dcfg.data_dir / dcfg.train_file,
        tokenizer,
        max_length=mcfg.max_seq_length,
        l1_to_idx=l1_to_idx,
        l2_to_idx=l2_to_idx,
    )
    val_ds = NewsClassificationDataset(
        dcfg.data_dir / dcfg.val_file,
        tokenizer,
        max_length=mcfg.max_seq_length,
        l1_to_idx=l1_to_idx,
        l2_to_idx=l2_to_idx,
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ──────────────────────────────────────────
    model = load_finbert_classifier(
        pretrained_model=mcfg.pretrained_model,
        num_level1=hcfg.num_level1,
        num_level2=hcfg.num_level2,
        num_sentiment=hcfg.num_sentiment,
        dropout=mcfg.dropout,
        alpha=lcfg.alpha,
        beta=lcfg.beta,
        gamma=lcfg.gamma,
    )
    model.to(device)

    # ── Optimizer & Scheduler ──────────────────────────
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": tcfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(params, lr=tcfg.learning_rate)

    total_steps = len(train_loader) * tcfg.num_epochs // tcfg.grad_accum_steps
    warmup_steps = int(total_steps * tcfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Mixed precision ────────────────────────────────
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg.fp16 and device.type == "cuda")

    # ── Training loop ──────────────────────────────────
    ocfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_l2_acc = 0.0
    global_step = 0

    for epoch in range(tcfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=tcfg.fp16 and device.type == "cuda"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    l1_label=batch["l1_label"],
                    l2_label=batch["l2_label"],
                    sentiment_label=batch["sentiment_label"],
                )
                loss = outputs["loss"] / tcfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % tcfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += outputs["loss"].item()

            if global_step > 0 and global_step % ocfg.log_steps == 0:
                avg = epoch_loss / (step + 1)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  [epoch {epoch + 1}/{tcfg.num_epochs}] step {global_step}, "
                    f"loss={avg:.4f}, l1_loss={outputs['l1_loss'].item():.4f}, "
                    f"l2_loss={outputs['l2_loss'].item():.4f}, "
                    f"sent_loss={outputs['sentiment_loss'].item():.4f}, lr={lr:.2e}"
                )
                wb.log_metrics(
                    {
                        "train/loss": avg,
                        "train/l1_loss": outputs["l1_loss"].item(),
                        "train/l2_loss": outputs["l2_loss"].item(),
                        "train/sentiment_loss": outputs["sentiment_loss"].item(),
                        "train/lr": lr,
                    },
                    step=global_step,
                )

            # ── Eval & Save checkpoints ────────────────
            if global_step > 0 and global_step % ocfg.eval_steps == 0:
                val_metrics = evaluate(model, val_loader, device)
                print(
                    f"  [eval step {global_step}] val_loss={val_metrics['loss']:.4f}, "
                    f"l1_acc={val_metrics['l1_accuracy']:.4f}, l2_acc={val_metrics['l2_accuracy']:.4f}, "
                    f"sent_acc={val_metrics['sentiment_accuracy']:.4f}"
                )
                wb.log_metrics(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/l1_accuracy": val_metrics["l1_accuracy"],
                        "val/l2_accuracy": val_metrics["l2_accuracy"],
                        "val/sentiment_accuracy": val_metrics["sentiment_accuracy"],
                    },
                    step=global_step,
                )
                if val_metrics["l2_accuracy"] > best_val_l2_acc:
                    best_val_l2_acc = val_metrics["l2_accuracy"]
                    save_dir = ocfg.output_dir / "best"
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    print(f"  ✓ New best model saved (l2_acc={best_val_l2_acc:.4f})")
                model.train()

            if global_step > 0 and global_step % ocfg.save_steps == 0:
                ckpt_dir = ocfg.output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

        # ── End of epoch ───────────────────────────────
        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{tcfg.num_epochs} done in {elapsed:.1f}s — "
            f"train_loss={avg_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_l1_acc={val_metrics['l1_accuracy']:.4f}, val_l2_acc={val_metrics['l2_accuracy']:.4f}, "
            f"val_sent_acc={val_metrics['sentiment_accuracy']:.4f}"
        )
        wb.log_metrics(
            {
                "epoch/train_loss": avg_loss,
                "epoch/val_loss": val_metrics["loss"],
                "epoch/val_l1_accuracy": val_metrics["l1_accuracy"],
                "epoch/val_l2_accuracy": val_metrics["l2_accuracy"],
                "epoch/val_sentiment_accuracy": val_metrics["sentiment_accuracy"],
            },
            step=global_step,
        )
        if val_metrics["l2_accuracy"] > best_val_l2_acc:
            best_val_l2_acc = val_metrics["l2_accuracy"]
            save_dir = ocfg.output_dir / "best"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  ✓ New best model saved (l2_acc={best_val_l2_acc:.4f})")

    # ── Save final model & label maps ──────────────────
    final_dir = ocfg.output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    label_info = {
        "l1_to_idx": l1_to_idx,
        "idx_to_l1": {str(k): v for k, v in idx_to_l1.items()},
        "l2_to_idx": l2_to_idx,
        "idx_to_l2": {str(k): v for k, v in idx_to_l2.items()},
    }
    with open(ocfg.output_dir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)

    # ── Final W&B summary & confusion matrix ───────────────────────────
    final_val = evaluate(model, val_loader, device)
    wb.log_summary(
        {
            "best_val_l2_accuracy": best_val_l2_acc,
            "final_val_l1_accuracy": final_val["l1_accuracy"],
            "final_val_l2_accuracy": final_val["l2_accuracy"],
            "final_val_sentiment_accuracy": final_val["sentiment_accuracy"],
        }
    )
    if cfg.wandb.log_l2_cm:
        l2_names = [idx_to_l2[i] for i in range(len(idx_to_l2))]
        wb.log_confusion_matrix(
            final_val["_l2_true"],
            final_val["_l2_pred"],
            class_names=l2_names,
            title="Level-2 Confusion Matrix",
        )
    wb.finish()

    print(f"\nTraining complete. Best val L2 accuracy: {best_val_l2_acc:.4f}")
    print(f"Checkpoints: {ocfg.output_dir}")
