"""Download MASSIVE 1.1 (CC BY 4.0) and extract the locales used here."""

import argparse
import tarfile

import _bootstrap  # noqa: F401
import requests

from weather_nlp.config import DATA_DIR, LOCALES, MASSIVE_URL, RAW_MASSIVE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if RAW_MASSIVE.exists() and not args.force:
        print(f"MASSIVE already present at {RAW_MASSIVE}")
        return
    raw = DATA_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tgz = raw / "massive.tar.gz"
    if not tgz.exists() or args.force:
        print(f"Downloading {MASSIVE_URL}")
        with requests.get(MASSIVE_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tgz, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    wanted = {f"1.1/data/{loc}.jsonl" for loc in LOCALES.values()} | {"1.1/LICENSE", "1.1/NOTICE.md"}
    with tarfile.open(tgz) as tar:
        members = [m for m in tar.getmembers() if m.name in wanted]
        tar.extractall(raw, members=members, filter="data")
    print(f"Extracted {len(members)} files to {RAW_MASSIVE.parent}")


if __name__ == "__main__":
    main()
