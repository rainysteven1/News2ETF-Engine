"""
Build a hierarchical ETF tracking-index industry dictionary via ZhipuAI.

Produces data/industry_dict.json with structure:
  { "大类": { "子类": ["指数名称", ...], ... }, ... }
"""

import json
import time

import polars as pl
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from zhipuai import ZhipuAI

from src.common import CONFIGS_DIR, DATA_DIR

PARQUET_FILE = DATA_DIR / "converted" / "主题ETF信息表-快照1_主题ETF.parquet"
OUTPUT_FILE = DATA_DIR / "industry_dict.json"

BATCH_SIZE = 50
DEFAULT_MODEL = "glm-4.7"


def _load_system_prompt() -> str:
    return (CONFIGS_DIR / "prompts" / "build_industry_system.txt").read_text(encoding="utf-8").strip()


def get_unique_index_names() -> list[str]:
    df = pl.read_parquet(PARQUET_FILE)
    df = df.filter(pl.col("跟踪指数名称") != "fund_trackindexname")
    return df["跟踪指数名称"].drop_nulls().unique().sort().to_list()


def _parse_llm_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


def classify_batch(client: ZhipuAI, names: list[str], model: str) -> dict[str, dict[str, list[str]]]:
    system_prompt = _load_system_prompt()
    names_str = "\n".join(f"- {n}" for n in names)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请将以下跟踪指数名称归类到两级分类体系中：\n{names_str}"},
        ],
        temperature=0.1,
    )
    return _parse_llm_json(response.choices[0].message.content.strip())  # type: ignore


def merge_dicts(
    base: dict[str, dict[str, list[str]]], new: dict[str, dict[str, list[str]]]
) -> dict[str, dict[str, list[str]]]:
    for major, subs in new.items():
        if major not in base:
            base[major] = {}
        for sub, names in subs.items():
            if sub in base[major]:
                base[major][sub].extend(names)
            else:
                base[major][sub] = names
    return base


def build_industry_dict(api_key: str, console: Console, model: str = DEFAULT_MODEL) -> None:
    client = ZhipuAI(api_key=api_key)

    console.print(f"[bold cyan]读取跟踪指数名称...[/bold cyan]  模型: [magenta]{model}[/magenta]")
    all_names = get_unique_index_names()
    total_batches = -(-len(all_names) // BATCH_SIZE)
    console.print(f"共 [bold]{len(all_names)}[/bold] 个不重复指数名称，分 [bold]{total_batches}[/bold] 批处理")

    industry_dict: dict[str, dict[str, list[str]]] = {}
    failed_batches: list[int] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("分类中...", total=total_batches)

        for i in range(0, len(all_names), BATCH_SIZE):
            batch = all_names[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            try:
                result = classify_batch(client, batch, model)
                industry_dict = merge_dicts(industry_dict, result)
            except Exception as e:
                progress.console.print(f"  [red]✗ 第 {batch_num} 批失败: {e}[/red]")
                failed_batches.append(batch_num)

            progress.advance(task)

            if i + BATCH_SIZE < len(all_names):
                time.sleep(1)

    # sort for stable output
    industry_dict = {
        major: {sub: sorted(names) for sub, names in sorted(subs.items())}
        for major, subs in sorted(industry_dict.items())
    }

    OUTPUT_FILE.write_text(
        json.dumps(industry_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_indices = sum(len(names) for subs in industry_dict.values() for names in subs.values())
    sub_count = sum(len(subs) for subs in industry_dict.values())

    console.print(
        f"\n[bold green]✓ 完成！[/bold green] 共 [bold]{len(industry_dict)}[/bold] 个大类，"
        f"[bold]{sub_count}[/bold] 个子类，覆盖 [bold]{total_indices}[/bold] 个指数名称"
    )
    if failed_batches:
        console.print(f"[yellow]⚠ 以下批次失败（可重跑）: {failed_batches}[/yellow]")
    console.print(f"已保存至: [cyan]{OUTPUT_FILE}[/cyan]")
