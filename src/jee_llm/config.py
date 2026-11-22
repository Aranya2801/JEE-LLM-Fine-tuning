"""YAML config + `key.sub=value` command line overrides.

    python -m jee_llm.train.sft configs/sft_qlora.yaml training.learning_rate=1e-4 model.name=Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _coerce(value: str) -> Any:
    # let yaml decide: "1e-4" -> float, "true" -> bool, "[a, b]" -> list, "null" -> None.
    # except YAML 1.1 also reads no/yes/on/off as bools, which breaks eval_strategy=no
    if value.strip().lower() in ("no", "yes", "on", "off", "y", "n"):
        return value.strip()
    out = yaml.safe_load(value)
    # and PyYAML reads "1e-4" as a *string* (it wants "1.0e-4"), which then
    # reaches the trainer as lr="1e-4". catch that here.
    if isinstance(out, str):
        try:
            return float(out)
        except ValueError:
            pass
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} should look like key.sub=value")
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"can't set {key}: {p} is not a mapping")
        node[leaf] = _coerce(value)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return apply_overrides(cfg, overrides or [])


def parse_cli(argv: list[str]) -> dict:
    if not argv or argv[0] in ("-h", "--help"):
        raise SystemExit("usage: <config.yaml> [key.sub=value ...]")
    return load_config(argv[0], argv[1:])
