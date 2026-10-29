"""Fine-tune the joint intent + slot model (optionally with a dialect head) on MASSIVE (+ dialect data).

Examples
  # full run (GPU / Colab recommended): MuRIL on en+hi+ta with dialect augmentation
  python scripts/train_nlu.py --encoder google/muril-base-cased --name muril_dialect

  # no-augmentation baseline for the ablation
  python scripts/train_nlu.py --encoder google/muril-base-cased --name muril_std --no-dialects

  # quick CPU run on a subset
  python scripts/train_nlu.py --name quick --max-per-file 1500 --epochs 1
"""

import argparse

import _bootstrap  # noqa: F401

from weather_nlp.config import LANGUAGES, MODELS_DIR
from weather_nlp.nlu.data import load_rows
from weather_nlp.nlu.train import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="google/muril-base-cased", help="e.g. google/muril-base-cased, xlm-roberta-base")
    ap.add_argument("--name", required=True)
    ap.add_argument("--langs", nargs="+", default=LANGUAGES)
    ap.add_argument("--no-dialects", action="store_true", help="train on standard MASSIVE only (ablation baseline)")
    ap.add_argument("--dialect-head", action="store_true", help="multi-task: also predict the dialect")
    ap.add_argument("--scenarios", nargs="*", help="restrict to MASSIVE scenarios, e.g. weather datetime")
    ap.add_argument("--max-per-file", type=int, help="subsample each source file (for quick CPU runs)")
    ap.add_argument("--max-dev-per-file", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--freeze-embeddings", action="store_true", help="freeze word embeddings (much faster on CPU)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--threads", type=int)
    args = ap.parse_args()

    use_dialects = not args.no_dialects
    train_rows = load_rows(args.langs, "train", use_dialects, args.scenarios, args.max_per_file, args.seed)
    dev_rows = load_rows(args.langs, "dev", use_dialects, args.scenarios, args.max_dev_per_file, args.seed)
    train(
        train_rows,
        dev_rows,
        encoder=args.encoder,
        out_dir=MODELS_DIR / "nlu" / args.name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        dialect_head=args.dialect_head,
        freeze_embeddings=args.freeze_embeddings,
        seed=args.seed,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
