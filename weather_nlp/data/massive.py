"""Loading MASSIVE and converting its bracketed annotations to token-level BIO tags."""

import json
import re
from pathlib import Path

from ..config import LOCALES, MASSIVE_DIR, RAW_MASSIVE
from ..preprocessing import tokenize

SLOT_PATTERN = re.compile(r"\[\s*([A-Za-z_]+)\s*:\s*([^\]]*?)\s*\]")


def parse_annot_utt(annot_utt: str):
    """'मुझे [date : कल] का मौसम बताओ' -> (tokens, BIO tags)."""
    tokens, tags = [], []
    pos = 0
    for m in SLOT_PATTERN.finditer(annot_utt):
        for t in tokenize(annot_utt[pos : m.start()]):
            tokens.append(t)
            tags.append("O")
        slot_type = m.group(1)
        for i, t in enumerate(tokenize(m.group(2))):
            tokens.append(t)
            tags.append(("B-" if i == 0 else "I-") + slot_type)
        pos = m.end()
    for t in tokenize(annot_utt[pos:]):
        tokens.append(t)
        tags.append("O")
    return tokens, tags


def to_annot_utt(tokens, tags) -> str:
    """Inverse of ``parse_annot_utt``."""
    parts, i = [], 0
    while i < len(tokens):
        if tags[i] == "O":
            parts.append(tokens[i])
            i += 1
            continue
        slot = tags[i][2:]
        j = i + 1
        while j < len(tokens) and tags[j] == f"I-{slot}":
            j += 1
        parts.append(f"[{slot} : {' '.join(tokens[i:j])}]")
        i = j
    return " ".join(parts)


def read_raw_locale(locale: str, raw_dir: Path = RAW_MASSIVE):
    with open(raw_dir / f"{locale}.jsonl", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def convert_record(rec: dict, lang: str) -> dict:
    tokens, tags = parse_annot_utt(rec["annot_utt"])
    return {
        "id": f"{lang}-{rec['id']}",
        "massive_id": rec["id"],
        "lang": lang,
        "dialect": "standard",
        "source": "massive",
        "partition": rec["partition"],
        "scenario": rec["scenario"],
        "intent": rec["intent"],
        "text": " ".join(tokens),
        "tokens": tokens,
        "tags": tags,
    }


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def processed_path(lang: str, split: str, base: Path = MASSIVE_DIR) -> Path:
    return base / f"{lang}_{split}.jsonl"


def load_split(lang: str, split: str, base: Path = MASSIVE_DIR):
    return read_jsonl(processed_path(lang, split, base))


def prepare_massive(languages=None, raw_dir: Path = RAW_MASSIVE, out_dir: Path = MASSIVE_DIR):
    stats = {}
    for lang in languages or LOCALES:
        by_split = {"train": [], "dev": [], "test": []}
        for rec in read_raw_locale(LOCALES[lang], raw_dir):
            by_split[rec["partition"]].append(convert_record(rec, lang))
        for split, rows in by_split.items():
            write_jsonl(processed_path(lang, split, out_dir), rows)
        stats[lang] = {k: len(v) for k, v in by_split.items()}
    return stats
