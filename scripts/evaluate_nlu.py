"""Per-dialect NLU evaluation and the dialect-robustness ablation.

Conditions (one row each in the output table):
  <model>             model on the raw (dialect) input
  <model>+norm        dialect ID -> lexicon normaliser -> model   (normalise-then-understand)
  <model>+norm-oracle same, with the gold dialect label (upper bound for the normaliser)

Slot scores: span F1 compares BIO spans on the input tokens (only defined for raw input);
value F1 compares (slot, value) pairs and is used to compare raw vs normalised input. For
normalised input the gold values are those of the standard source sentence.

Examples
  python scripts/evaluate_nlu.py --models rules muril_std muril_dialect --normalize
  python scripts/evaluate_nlu.py --models muril_dialect --scope all
"""

import argparse
import json
import time
from collections import defaultdict

import _bootstrap  # noqa: F401

from weather_nlp.config import DIALECTS, MODELS_DIR, PROCESSED, REPORTS_DIR, WEATHER_INTENT, dialect_label
from weather_nlp.data.massive import load_split, read_jsonl
from weather_nlp.dialect import LexiconNormalizer
from weather_nlp.dialect.identifier import DialectIdentifier
from weather_nlp.nlu.data import dialect_path
from weather_nlp.dialogue import Canonicalizer
from weather_nlp.dialogue.canonicalize import SLOT_KEYS
from weather_nlp.evaluation.followup import slot_key
from weather_nlp.nlu.metrics import nlu_report, slot_values
from weather_nlp.nlu.predictor import RuleNLU, TransformerNLU, spans_from_tags


def load_test(scope, langs, include_real=True):
    rows = []
    for lang in langs:
        rows += load_split(lang, "test")
        for d in DIALECTS[lang]:
            p = dialect_path(lang, d, "test")
            if d != "standard" and p.exists():
                rows += read_jsonl(p)
    if include_real and (PROCESSED / "dialect_real").exists():
        for p in (PROCESSED / "dialect_real").glob("*.jsonl"):
            rows += [dict(r, dialect=r["dialect"] + "(real)") for r in read_jsonl(p) if r["lang"] in langs]
    if scope == "weather":
        rows = [r for r in rows if r["scenario"] == "weather"]
    return rows


def load_nlu(name):
    if name == "rules":
        return RuleNLU()
    return TransformerNLU(MODELS_DIR / "nlu" / name)


def canonical_scores(pairs, canon):
    """Task-level slot evaluation on language-independent values (place key, date offset, ...).

    frame_acc: intent and all canonical slots correct. canon_f1: micro F1 over non-null canonical slots.
    Unlike surface metrics this is invariant to normalisation choosing a different but equivalent form.
    """
    tp = n_pred = n_gold = frame_ok = 0
    for r, p in pairs:
        g = canon.canonicalize({"slots": spans_from_tags(r["tokens"], r["tags"])}, r["tokens"])
        q = canon.canonicalize({"slots": spans_from_tags(p["_tokens"], p["tags"])}, p["_tokens"])
        gk = {k: slot_key(k, g[k]) for k in SLOT_KEYS if g[k] is not None}
        pk = {k: slot_key(k, q[k]) for k in SLOT_KEYS if q[k] is not None}
        tp += sum(pk.get(k) == v for k, v in gk.items())
        n_pred += len(pk)
        n_gold += len(gk)
        frame_ok += r["intent"] == p["intent"] and gk == pk
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    return {"canon_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "frame_acc": frame_ok / max(len(pairs), 1)}


def score(rows, preds, raw: bool):
    canon = Canonicalizer(backoff=False)
    groups = defaultdict(list)
    for r, p in zip(rows, preds):
        groups["ALL"].append((r, p))
        groups[dialect_label(r["lang"], r["dialect"])].append((r, p))
        if r["dialect"] != "standard":
            groups[f"{r['lang']}:ALL-DIALECTS"].append((r, p))
    out = {}
    for g, pairs in groups.items():
        rs, ps = zip(*pairs)
        if raw:
            gold_vals = [slot_values(r["tokens"], r["tags"]) for r in rs]
        else:
            gold_vals = [
                slot_values(r["standard_tokens"], r["standard_tags"]) if r.get("standard_tokens") else slot_values(r["tokens"], r["tags"])
                for r in rs
            ]
        out[g] = nlu_report(
            [r["intent"] for r in rs],
            [p["intent"] for p in ps],
            [r["tags"] for r in rs] if raw else None,
            [p["tags"] for p in ps] if raw else None,
            gold_vals,
            [slot_values(p["_tokens"], p["tags"]) for p in ps],
        )
        out[g].update(canonical_scores(pairs, canon))
    return out


def run(nlu, rows, normalize=None, dialect_id=None, normalizer=None):
    inputs = []
    for r in rows:
        toks = r["tokens"]
        if normalize:
            if normalize == "oracle":
                d = r["dialect"].replace("(real)", "")
            else:
                d = dialect_id.predict_one(r["text"])[0].split(":")[1]
            toks = normalizer.normalize(toks, r["lang"], d)
        inputs.append(toks)
    preds = nlu.predict_batch(inputs)
    for p, t in zip(preds, inputs):
        p["_tokens"] = t
    return preds


def md_table(results, metric_cols):
    groups = sorted({g for res in results.values() for g in res})
    lines = ["| condition | group | n | " + " | ".join(metric_cols) + " |", "|" + "---|" * (3 + len(metric_cols))]
    for cond, res in results.items():
        for g in groups:
            if g in res:
                r = res[g]
                vals = [f"{r[m] * 100:.1f}" if m in r else "-" for m in metric_cols]
                lines.append(f"| {cond} | {g} | {r['n']} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["rules"])
    ap.add_argument("--scope", choices=["weather", "all"], default="weather")
    ap.add_argument("--langs", nargs="+", default=["en", "hi", "ta"])
    ap.add_argument("--normalize", action="store_true", help="add normalise-then-NLU conditions")
    ap.add_argument("--dialect-model", default=str(MODELS_DIR / "dialect_id" / "svm.joblib"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = load_test(args.scope, args.langs)
    print(f"{len(rows)} test utterances (scope={args.scope})")
    normalizer = LexiconNormalizer()
    dialect_id = DialectIdentifier.load(args.dialect_model) if args.normalize else None

    results = {}
    for name in args.models:
        nlu = load_nlu(name)
        t0 = time.time()
        results[name] = score(rows, run(nlu, rows), raw=True)
        print(f"[{name}] raw done in {time.time() - t0:.0f}s")
        if args.normalize:
            results[f"{name}+norm"] = score(rows, run(nlu, rows, "classifier", dialect_id, normalizer), raw=False)
            results[f"{name}+norm-oracle"] = score(rows, run(nlu, rows, "oracle", None, normalizer), raw=False)
            print(f"[{name}] normalised conditions done")

    cols = ["intent_acc", "slot_f1", "exact_match", "value_f1", "canon_f1", "frame_acc"]
    table = md_table(results, cols)
    print(table)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"nlu_eval_{args.scope}{('_' + args.tag) if args.tag else ''}"
    (REPORTS_DIR / f"{stem}.md").write_text(table + "\n", encoding="utf-8")
    with open(REPORTS_DIR / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"saved reports/{stem}.md / .json")


if __name__ == "__main__":
    main()
