"""Interactive multilingual weather assistant.

  python scripts/chat.py --nlu muril_dialect            # recommended (no dialect normaliser)
  python scripts/chat.py --nlu muril_std --normalize    # ablation: normalise-then-understand
  python scripts/chat.py --nlu rules --offline --demo   # no model / no internet

Commands inside the chat:
  /lang en|hi|ta|auto   reply language (auto = same as the question)
  /audio <file.wav>     ask by voice (Whisper; needs `pip install soundfile`)
  /state  /debug  /reset  /quit
"""

import argparse
import json
import sys

import _bootstrap  # noqa: F401

from weather_nlp.assistant import WeatherAssistant
from weather_nlp.config import MODELS_DIR

# Hindi/Tamil print cleanly on Windows consoles (cp1252 otherwise crashes).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEMO = [
    ("बेंगलुरु में कल मौसम कैसा रहेगा?", None),
    ("சென்னையில்?", None),
    ("நாளைக்கு கோவைல மழை வருமுங்களா?", None),
    ("और परसों?", "ta"),
    ("what about Madurai", "hi"),
    ("काल्ह बलिया में बरखा होई का?", None),
    ("अर रोहतक म्ह?", None),
    ("set an alarm for 6 am", None),
]


def show(res, debug):
    print(f"  [{res['lang']}:{res['dialect']} ({res['dialect_conf']:.2f}) | intent {res['nlu']['intent']} ({res['nlu']['intent_conf']:.2f}) | {res['mode']}]")
    if debug:
        if res["normalized"] is not None:
            print("  normalized:", " ".join(res["normalized"]))
        print("  slots:", [(s["type"], s["value"]) for s in res["nlu"]["slots"]])
        print("  state:", json.dumps(res["state"], ensure_ascii=False))
    print("bot>", res["response"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nlu", default="muril_dialect", help="model name under models/nlu, or 'rules'")
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="optional lexicon dialect->standard rewrite before NLU (helps muril_std; not needed for muril_dialect)",
    )
    ap.add_argument("--offline", action="store_true", help="use mock weather instead of Open-Meteo")
    ap.add_argument("--reply-lang", choices=["en", "hi", "ta"])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--demo", action="store_true", help="run a scripted cross-lingual, cross-dialect conversation")
    args = ap.parse_args()

    nlu_dir = None if args.nlu == "rules" else MODELS_DIR / "nlu" / args.nlu
    if nlu_dir and not nlu_dir.exists():
        print(f"(model {nlu_dir} not found; falling back to the rule-based NLU)")
        nlu_dir = None
    bot = WeatherAssistant.from_dirs(nlu_dir, normalize=args.normalize, offline=args.offline, reply_lang=args.reply_lang)
    print(
        f"NLU: {bot.nlu.name} | dialect ID: {'on' if bot.dialect_id else 'off'} | "
        f"normalize: {'on' if args.normalize else 'off'} | weather: {'mock' if args.offline else 'Open-Meteo'}"
    )
    debug, reply_lang = args.debug, args.reply_lang

    if args.demo:
        for text, lang in DEMO:
            print(f"\nyou> {text}" + (f"   (reply in {lang})" if lang else ""))
            show(bot.handle(text, reply_lang=lang), True)
        return

    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            if cmd == "/quit":
                break
            if cmd == "/reset":
                bot.reset()
                print("(context cleared)")
            elif cmd == "/state":
                print(json.dumps(bot.context.state.snapshot(), ensure_ascii=False, indent=1))
            elif cmd == "/debug":
                debug = not debug
            elif cmd == "/lang":
                reply_lang = None if arg in ("", "auto") else arg
                print(f"(reply language: {reply_lang or 'same as question'})")
            elif cmd == "/audio":
                from weather_nlp.asr import transcribe

                text = transcribe(arg.strip())
                print(f"(heard) {text}")
                show(bot.handle(text, reply_lang=reply_lang), debug)
            continue
        show(bot.handle(text, reply_lang=reply_lang), debug)


if __name__ == "__main__":
    main()
