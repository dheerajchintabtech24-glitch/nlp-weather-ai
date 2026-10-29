"""Cross-lingual / cross-dialect follow-up evaluation of the conversation-state manager.

Reports joint goal accuracy (JGA) after the follow-up turn, plus intent, changed-slot and carried-slot
accuracy, broken down by condition (same-standard is the control), by follow-up variety and by
language pair. Rows:
  <model>               end-to-end (turn 1 predicted)
  <model>/oracle-turn1  turn 1 state = gold (isolates the follow-up turn)
  <model>/no-context    carry-over disabled (lower bound)

  python scripts/evaluate_context.py --models rules muril_dialect
"""

import argparse
import json

import _bootstrap  # noqa: F401

from weather_nlp.config import FOLLOWUP_DIR, MODELS_DIR, REPORTS_DIR
from weather_nlp.data.massive import read_jsonl, write_jsonl
from weather_nlp.dialect import LexiconNormalizer
from weather_nlp.dialect.identifier import DialectIdentifier
from weather_nlp.evaluation.followup import build_items, evaluate
from weather_nlp.nlu.predictor import RuleNLU, TransformerNLU


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["rules"])
    ap.add_argument("--n-turn1", type=int, default=150, help="turn-1 utterances per language")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--normalize", action="store_true", help="also evaluate with dialect ID + normaliser on turn 2")
    ap.add_argument("--no-backoff", action="store_true", help="disable lexicon/gazetteer slot backoff")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    path = FOLLOWUP_DIR / f"followup_{args.n_turn1}_{args.seed}.jsonl"
    if path.exists() and not args.rebuild:
        items = read_jsonl(path)
    else:
        items = build_items(args.n_turn1, args.seed)
        write_jsonl(path, items)
    print(f"{len(items)} follow-up conversations ({path})")

    backoff = not args.no_backoff
    results = {}
    for name in args.models:
        nlu = RuleNLU() if name == "rules" else TransformerNLU(MODELS_DIR / "nlu" / name)
        results[name] = evaluate(items, nlu, backoff=backoff)
        results[f"{name}/oracle-turn1"] = evaluate(items, nlu, oracle_turn1=True, backoff=backoff)
        results[f"{name}/no-context"] = evaluate(items, nlu, carry_over=False, backoff=backoff)
        if args.normalize:
            dia = DialectIdentifier.load(MODELS_DIR / "dialect_id" / "svm.joblib")
            results[f"{name}/norm"] = evaluate(items, nlu, backoff=backoff, normalizer=LexiconNormalizer(), dialect_id=dia)
        print(f"[{name}] done")

    order = ["ALL", "same-standard", "cross-standard", "same-dialect", "cross-dialect"]
    cols = ["jga", "intent_acc", "target_acc", "carry_acc", "followup_detected"]
    lines = ["| system | group | n | " + " | ".join(cols) + " |", "|" + "---|" * (3 + len(cols))]
    for sysname, res in results.items():
        groups = order + sorted(g for g in res if g not in order)
        for g in groups:
            if g in res:
                r = res[g]
                lines.append(f"| {sysname} | {g} | {int(r['n'])} | " + " | ".join(f"{r[c] * 100:.1f}" for c in cols) + " |")
    table = "\n".join(lines)
    print(table)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "" if backoff else "_nobackoff"
    (REPORTS_DIR / f"context_eval{tag}.md").write_text(table + "\n", encoding="utf-8")
    with open(REPORTS_DIR / f"context_eval{tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
