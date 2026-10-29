"""Assembling NLU training/evaluation sets from processed MASSIVE and dialect files."""

import random
from pathlib import Path

from ..config import DIALECT_DIR, DIALECTS, MASSIVE_DIR, dialect_label
from ..data.massive import read_jsonl


def dialect_path(lang: str, dialect: str, split: str, base: Path = DIALECT_DIR) -> Path:
    return base / f"{lang}_{dialect}_{split}.jsonl"


def load_rows(langs, split, include_dialects=True, scenarios=None, max_per_file=None, seed=13, keep_scenarios=("weather",)):
    """``max_per_file`` subsamples each file but always keeps rows of ``keep_scenarios``."""
    rng = random.Random(seed)
    rows = []
    files = [MASSIVE_DIR / f"{lang}_{split}.jsonl" for lang in langs]
    if include_dialects:
        files += [dialect_path(l, d, split) for l in langs for d in DIALECTS[l] if d != "standard"]
    for path in files:
        if not path.exists():
            continue
        part = read_jsonl(path)
        if scenarios:
            part = [r for r in part if r["scenario"] in scenarios]
        if max_per_file and len(part) > max_per_file:
            keep = [r for r in part if r["scenario"] in keep_scenarios]
            rest = [r for r in part if r["scenario"] not in keep_scenarios]
            part = keep + rng.sample(rest, max(0, min(len(rest), max_per_file - len(keep))))
        rows += part
    return rows


def build_label_maps(rows, include_dialect_head=False):
    intents = sorted({r["intent"] for r in rows})
    slot_types = sorted({t[2:] for r in rows for t in r["tags"] if t != "O"})
    slots = ["O"] + [f"{p}-{s}" for s in slot_types for p in ("B", "I")]
    dialects = None
    if include_dialect_head:
        dialects = [dialect_label(l, d) for l in DIALECTS for d in DIALECTS[l]]
    return intents, slots, dialects
