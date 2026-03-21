"""
Hierarchy label system — loads configs/hierarchy.yaml and builds label mappings.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_HIERARCHY_PATH = _ROOT / "configs" / "hierarchy.yaml"


def load_hierarchy(path: Path = _HIERARCHY_PATH) -> dict[str, list[str]]:
    """Return {level1_name: [level2_name, ...]} from YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_label_maps(
    hierarchy: dict[str, list[str]] | None = None,
) -> tuple[dict[str, int], dict[int, str], dict[str, int], dict[int, str], dict[str, str]]:
    """Build all label ↔ index mappings.

    Returns:
        l1_to_idx:   {"科技信息": 0, "高端制造": 1, ...}
        idx_to_l1:   {0: "科技信息", ...}
        l2_to_idx:   {"半导体/芯片": 0, "人工智能": 1, ...}  (global index)
        idx_to_l2:   {0: "半导体/芯片", ...}
        l2_to_l1:    {"半导体/芯片": "科技信息", ...}
    """
    if hierarchy is None:
        hierarchy = load_hierarchy()

    l1_to_idx: dict[str, int] = {}
    idx_to_l1: dict[int, str] = {}
    l2_to_idx: dict[str, int] = {}
    idx_to_l2: dict[int, str] = {}
    l2_to_l1: dict[str, str] = {}

    l2_counter = 0
    for l1_idx, (l1_name, l2_list) in enumerate(hierarchy.items()):
        l1_to_idx[l1_name] = l1_idx
        idx_to_l1[l1_idx] = l1_name
        for l2_name in l2_list:
            l2_to_idx[l2_name] = l2_counter
            idx_to_l2[l2_counter] = l2_name
            l2_to_l1[l2_name] = l1_name
            l2_counter += 1

    return l1_to_idx, idx_to_l1, l2_to_idx, idx_to_l2, l2_to_l1


def build_l1_to_l2_indices(
    hierarchy: dict[str, list[str]] | None = None,
) -> dict[int, list[int]]:
    """Build mapping: l1_idx -> [l2_global_idx, ...].

    Used for masking level-2 logits during inference:
    given a predicted L1, only activate the L2 slots belonging to that L1.
    """
    if hierarchy is None:
        hierarchy = load_hierarchy()

    l1_to_idx, _, l2_to_idx, _, _ = build_label_maps(hierarchy)
    result: dict[int, list[int]] = {}
    for l1_name, l2_list in hierarchy.items():
        l1_idx = l1_to_idx[l1_name]
        result[l1_idx] = [l2_to_idx[l2] for l2 in l2_list]
    return result
