"""Train and compare dialect identifiers (char n-gram TF-IDF + SVM / LogReg / Naive Bayes)."""

import argparse
import json
import random

import _bootstrap  # noqa: F401
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from weather_nlp.config import MODELS_DIR, PROCESSED, REPORTS_DIR, dialect_label
from weather_nlp.data.massive import load_split, read_jsonl
from weather_nlp.dialect.identifier import CLASSIFIERS, DialectIdentifier
from weather_nlp.nlu.data import load_rows


def standard_like(lang, split, fraction, seed):
    """Standard sentences with the same scenario mix as the dialect files (all weather + a fraction of the rest)."""
    rng = random.Random(f"std-{lang}-{split}-{seed}")
    return [r for r in load_split(lang, split) if r["scenario"] == "weather" or rng.random() < fraction]


def collect(split, fraction, seed):
    rows = []
    for lang in ("hi", "ta"):
        rows += standard_like(lang, split, fraction, seed)
        rows += [r for r in load_rows([lang], split, include_dialects=True) if r["dialect"] != "standard"]
    rows += standard_like("en", split, fraction, seed)
    return [r["text"] for r in rows], [dialect_label(r["lang"], r["dialect"]) for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classifiers", nargs="+", default=list(CLASSIFIERS))
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    Xtr, ytr = collect("train", 0.3, args.seed)
    Xte, yte = collect("test", 0.15, args.seed)
    real = []
    real_dir = PROCESSED / "dialect_real"
    if real_dir.exists():
        for p in real_dir.glob("*.jsonl"):
            real += read_jsonl(p)
    out_dir = REPORTS_DIR / "dialect_id"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.classifiers:
        model = DialectIdentifier(name).fit(Xtr, ytr)
        pred = [p for p, _ in model.predict(Xte)]
        labels = model.labels
        rep = classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0)
        summary[name] = {"macro_f1": f1_score(yte, pred, average="macro"), "accuracy": rep["accuracy"]}
        if real:
            rp = [p for p, _ in model.predict([r["text"] for r in real])]
            ry = [dialect_label(r["lang"], r["dialect"]) for r in real]
            summary[name]["real_accuracy"] = sum(a == b for a, b in zip(rp, ry)) / len(ry)
        cm = confusion_matrix(yte, pred, labels=labels)
        with open(out_dir / f"{name}_report.json", "w", encoding="utf-8") as f:
            json.dump({"labels": labels, "report": rep, "confusion": cm.tolist()}, f, indent=2)
        model.save(MODELS_DIR / "dialect_id" / f"{name}.joblib")
        print(f"\n=== {name}: macro-F1 {summary[name]['macro_f1']:.4f}, acc {summary[name]['accuracy']:.4f}")
        print(classification_report(yte, pred, labels=labels, digits=3, zero_division=0))
        print("confusion (rows=gold, cols=pred):", labels)
        for lab, row in zip(labels, cm):
            print(f"  {lab:12s} {' '.join(f'{v:4d}' for v in row)}")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
