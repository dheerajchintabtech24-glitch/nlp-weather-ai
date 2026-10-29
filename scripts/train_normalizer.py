"""Train the char-level seq2seq normaliser and compare normalisers on dialect -> standard test pairs.

Systems: identity (copy input), lexicon (gold dialect), lexicon (predicted dialect), seq2seq (gold
dialect), seq2seq (predicted dialect). Metrics: sentence exact match, character error rate (CER),
and chrF when sacrebleu is installed.
"""

import argparse
import json
import random

import _bootstrap  # noqa: F401

from weather_nlp.config import MODELS_DIR, REPORTS_DIR
from weather_nlp.dialect import LexiconNormalizer
from weather_nlp.dialect.identifier import DialectIdentifier
from weather_nlp.dialect.seq2seq import Seq2SeqNormalizer, train_seq2seq
from weather_nlp.nlu.data import load_rows
from weather_nlp.preprocessing import tokenize


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def metrics(hyps, refs):
    out = {
        "exact": sum(h == r for h, r in zip(hyps, refs)) / len(refs),
        "cer": sum(edit_distance(h, r) for h, r in zip(hyps, refs)) / max(1, sum(len(r) for r in refs)),
    }
    try:
        import sacrebleu

        out["chrf"] = sacrebleu.corpus_chrf(hyps, [refs]).score / 100
    except ImportError:
        pass
    return out


def pairs(split, max_per_file=None):
    rows = [r for r in load_rows(["hi", "ta"], split, True, max_per_file=max_per_file) if r["dialect"] != "standard"]
    return [(r["text"], r["standard_text"], r["lang"], r["dialect"]) for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--max-train", type=int, default=12000)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    out_dir = MODELS_DIR / "normalizer_seq2seq"
    train_p, dev_p, test_p = pairs("train"), pairs("dev", 60), pairs("test")
    if len(train_p) > args.max_train:
        train_p = random.Random(13).sample(train_p, args.max_train)
    print(f"pairs: train {len(train_p)} dev {len(dev_p)} test {len(test_p)}")
    s2s = (
        Seq2SeqNormalizer.load(out_dir)
        if args.skip_train
        else train_seq2seq(train_p, dev_p, out_dir, epochs=args.epochs, batch_size=args.batch_size, hidden=args.hidden)
    )

    srcs = [s for s, *_ in test_p]
    refs = [t for _, t, _, _ in test_p]
    langs = [l for *_, l, _ in test_p]
    gold_d = [d for *_, d in test_p]
    did = DialectIdentifier.load(MODELS_DIR / "dialect_id" / "svm.joblib")
    pred_d = [p.split(":")[1] for p, _ in did.predict(srcs)]
    lex = LexiconNormalizer()

    def lex_run(dialects):
        return [" ".join(lex.normalize(tokenize(s), l, d)) for s, l, d in zip(srcs, langs, dialects)]

    def s2s_run(dialects):
        out = s2s.normalize_texts(srcs, langs, [d if d != "standard" else "standard" for d in dialects])
        return [o if d != "standard" else s for o, d, s in zip(out, dialects, srcs)]

    results = {
        "identity": metrics(srcs, refs),
        "lexicon/gold-dialect": metrics(lex_run(gold_d), refs),
        "lexicon/pred-dialect": metrics(lex_run(pred_d), refs),
        "seq2seq/gold-dialect": metrics(s2s_run(gold_d), refs),
        "seq2seq/pred-dialect": metrics(s2s_run(pred_d), refs),
    }
    lines = ["| system | exact | CER | chrF |", "|---|---|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['exact'] * 100:.1f} | {v['cer'] * 100:.2f} | {v.get('chrf', float('nan')) * 100:.1f} |")
    print("\n".join(lines))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "normalizer_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS_DIR / "normalizer_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
