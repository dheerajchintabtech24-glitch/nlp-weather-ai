"""Convert MASSIVE to BIO, generate slot-preserving dialect data, and import collected real dialect data.

Outputs
  data/processed/massive/{lang}_{split}.jsonl            standard MASSIVE (en, hi, ta)
  data/processed/dialect/{lang}_{dialect}_{split}.jsonl  synthetic dialect rewrites (hi, ta)
  data/processed/dialect_real/{lang}_{dialect}.jsonl     native-speaker data from data/dialect/collected/*.csv
"""

import argparse
import csv
import random
from collections import Counter

import _bootstrap  # noqa: F401

from weather_nlp.config import COLLECTED_DIALECT_DIR, DIALECT_DIR, PROCESSED, WEATHER_INTENT
from weather_nlp.data.massive import load_split, parse_annot_utt, prepare_massive, write_jsonl
from weather_nlp.dialect import DialectAugmenter, DialectLexicon
from weather_nlp.nlu.data import dialect_path


def augment(args):
    lex = DialectLexicon()
    aug = DialectAugmenter(lex, p_replace=args.p_replace, p_particle=args.p_particle)
    fractions = {"train": args.train_fraction, "dev": args.eval_fraction, "test": args.eval_fraction}
    summary = Counter()
    for lang in ("hi", "ta"):
        for split, frac in fractions.items():
            rows = load_split(lang, split)
            for dialect in lex.dialects(lang):
                rng = random.Random(f"{lang}-{dialect}-{split}-{args.seed}")
                out = []
                for r in rows:
                    if r["scenario"] != "weather" and rng.random() >= frac:
                        continue
                    for k, (toks, tags) in enumerate(aug.variants(r, dialect, k=args.variants, rng=rng)):
                        out.append({
                            **r,
                            "id": f"{r['id']}-{dialect}-{k}",
                            "dialect": dialect,
                            "source": "synthetic",
                            "text": " ".join(toks),
                            "tokens": toks,
                            "tags": tags,
                            "standard_text": r["text"],
                            "standard_tokens": r["tokens"],
                            "standard_tags": r["tags"],
                        })
                write_jsonl(dialect_path(lang, dialect, split), out)
                summary[(lang, dialect, split)] = len(out)
    return summary


def import_collected():
    """CSV columns: lang,dialect,annot_utt,intent,standard_annot_utt[,region,annotator]."""
    out_dir = PROCESSED / "dialect_real"
    groups = {}
    for path in sorted(COLLECTED_DIALECT_DIR.glob("*.csv")):
        if path.name.lower().startswith("template"):
            continue
        with open(path, encoding="utf-8-sig", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if not row.get("annot_utt"):
                    continue
                toks, tags = parse_annot_utt(row["annot_utt"])
                std = row.get("standard_annot_utt") or ""
                stoks, stags = parse_annot_utt(std) if std else (None, None)
                rec = {
                    "id": f"real-{path.stem}-{i}",
                    "lang": row["lang"].strip(),
                    "dialect": row["dialect"].strip(),
                    "source": "collected",
                    "partition": "test",
                    "scenario": "weather" if (row.get("intent") or WEATHER_INTENT) == WEATHER_INTENT else row["intent"].split("_")[0],
                    "intent": row.get("intent") or WEATHER_INTENT,
                    "text": " ".join(toks),
                    "tokens": toks,
                    "tags": tags,
                    "standard_text": " ".join(stoks) if stoks else None,
                    "standard_tokens": stoks,
                    "standard_tags": stags,
                    "region": row.get("region"),
                    "annotator": row.get("annotator"),
                }
                groups.setdefault((rec["lang"], rec["dialect"]), []).append(rec)
    for (lang, dialect), rows in groups.items():
        write_jsonl(out_dir / f"{lang}_{dialect}.jsonl", rows)
    return {k: len(v) for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", type=int, default=1, help="dialect rewrites per source sentence")
    ap.add_argument("--train-fraction", type=float, default=0.3, help="share of non-weather train sentences to rewrite (weather: all)")
    ap.add_argument("--eval-fraction", type=float, default=0.15, help="share of non-weather dev/test sentences to rewrite")
    ap.add_argument("--p-replace", type=float, default=0.85)
    ap.add_argument("--p-particle", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    print("MASSIVE:", prepare_massive())
    DIALECT_DIR.mkdir(parents=True, exist_ok=True)
    for (lang, dialect, split), n in sorted(augment(args).items()):
        print(f"dialect {lang}:{dialect:9s} {split:5s} {n}")
    real = import_collected()
    print("collected real dialect data:", real or "none yet (see data/dialect/collected/TEMPLATE.csv)")


if __name__ == "__main__":
    main()
