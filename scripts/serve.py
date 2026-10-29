"""Presentation UI for the weather NLP assistant.

  python scripts/serve.py
  python scripts/serve.py --nlu muril_dialect --port 7860
  python scripts/serve.py --nlu rules --offline
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import _bootstrap  # noqa: F401

from weather_nlp.assistant import WeatherAssistant
from weather_nlp.config import DIALECTS, MODELS_DIR, REPORTS_DIR, ROOT
from weather_nlp.dialogue.canonicalize import SLOT_KEYS
from weather_nlp.nlg.responder import condition_group

WEB_DIR = ROOT / "web"
LOCK = threading.Lock()
ASR_LOCK = threading.Lock()
CANCELLED = set()
BOT: WeatherAssistant | None = None
ASR_MODEL = "openai/whisper-small"
META = {}


DEMO = [
    {
        "text": "बेंगलुरु में कल मौसम कैसा रहेगा?",
        "reply_lang": None,
        "title": "Standard Hindi · new query",
        "note": "Script ID reads Devanagari as Hindi. MuRIL jointly predicts weather_query with place=Bengaluru and date=tomorrow. The conversation state is created from scratch.",
    },
    {
        "text": "சென்னையில்?",
        "reply_lang": None,
        "title": "Tamil elliptical follow-up",
        "note": "Four tokens, no weather verb. Place becomes Chennai; date=tomorrow is carried across languages because the blackboard only stores canonical symbols.",
    },
    {
        "text": "நாளைக்கு கோவைல மழை வருமுங்களா?",
        "reply_lang": None,
        "title": "Kongu Tamil · rain in Coimbatore",
        "note": "Dialect ID should prefer Kongu (spoken locative கோவைல, particle -ங்க). Slots: place=Coimbatore, date=tomorrow, descriptor=rain.",
    },
    {
        "text": "और परसों?",
        "reply_lang": "ta",
        "title": "Hindi follow-up · reply in Tamil",
        "note": "Input language, dialect and reply language are independent. परसों updates the date; place carries over. The answer is rendered in Tamil from the same canonical state.",
    },
    {
        "text": "what about Madurai",
        "reply_lang": "hi",
        "title": "English follow-up · reply in Hindi",
        "note": "A short English place switch. Canonical place becomes Madurai; date still carries. Grounded NLG speaks Hindi.",
    },
    {
        "text": "काल्ह बलिया में बरखा होई का?",
        "reply_lang": None,
        "title": "Bhojpuri-influenced Hindi",
        "note": "काल्ह / बरखा / होई are Bhojpuri forms. The canonicaliser maps them onto the same date and rain symbols a standard Hindi sentence would produce.",
    },
    {
        "text": "अर रोहतक म्ह?",
        "reply_lang": None,
        "title": "Haryanvi elliptical follow-up",
        "note": "अर … म्ह is a Haryanvi follow-up frame. Place switches to Rohtak; rain and date are inherited from the Bhojpuri turn.",
    },
    {
        "text": "set an alarm for 6 am",
        "reply_lang": None,
        "title": "Out-of-scope intent",
        "note": "MuRIL was trained on all 60 MASSIVE intents, so it can reject non-weather requests instead of hallucinating a forecast.",
    },
]


def _jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def _slot_provenance(frame, state):
    out = {}
    slots = (state or {}).get("slots") or {}
    for key in SLOT_KEYS:
        now, fresh = slots.get(key), (frame or {}).get(key)
        if now is None:
            out[key] = "empty"
        elif fresh is not None:
            out[key] = "new"
        else:
            out[key] = "carried"
    return out


def _weather_view(tool):
    if not tool or tool.get("status") != "ok":
        return None
    p = tool.get("payload") or {}
    day = (p.get("days") or [None])[0]
    cur = p.get("current") or p.get("part") or {}
    code = cur.get("code") if cur else None
    if code is None and day:
        code = day.get("code")
    temp = cur.get("temp")
    if temp is None and day:
        temp = day.get("tmax")
    view = {
        "kind": p.get("kind"),
        "place": p.get("place"),
        "descriptor": p.get("descriptor"),
        "condition": condition_group(code) if code is not None else None,
        "code": code,
        "temp": temp,
        "tmax": day.get("tmax") if day else None,
        "tmin": day.get("tmin") if day else None,
        "pop": (cur.get("pop") if "pop" in cur else None) or (day.get("pop") if day else None),
        "humidity": cur.get("humidity"),
        "wind": cur.get("wind") or (day.get("wind") if day else None),
        "days": p.get("days") or [],
    }
    trip = p.get("trip")
    if p.get("kind") == "travel" and trip:
        origin = p.get("origin") or {}
        view["travel"] = {
            "km": trip.get("km"),
            "eta_min": trip.get("minutes"),
            "base_min": trip.get("base_minutes"),
            "factor": trip.get("factor"),
            "reason": trip.get("reason"),
            "source": trip.get("source"),
            "origin": (origin.get("names") or {}).get("en") or origin.get("canonical"),
        }
    return view


def pack(result: dict) -> dict:
    """JSON payload for the UI. Speech audio is fetched separately so /api/ask stays fast."""
    tool = result.get("tool") or {}
    speech_lang = result.get("speech_lang") or result.get("lang") or "en"
    speech_text = result.get("speech_text") or ""
    if not speech_text and result.get("response"):
        # Defensive: never leave the UI without a speakable string.
        from weather_nlp.speech_text import prepare_speech

        speech = prepare_speech(
            result["response"],
            speech_lang,
            result.get("dialect") or "standard",
            float(result.get("dialect_conf") or 0.0),
        )
        speech_lang = speech["speech_lang"]
        speech_text = speech["speech_text"]
        result = {**result, **speech}
    return _jsonable(
        {
            **result,
            "speech_lang": speech_lang,
            "speech_text": speech_text,
            "speech_base": result.get("speech_base") or speech_text,
            "speech_dialect": result.get("speech_dialect") or "standard",
            "provenance": _slot_provenance(result.get("frame"), result.get("state")),
            "weather": _weather_view(tool),
        }
    )


def _pct(x):
    return None if x is None else round(100 * float(x), 1)


def load_metrics():
    dialect = REPORTS_DIR / "dialect_id" / "summary.json"
    nlu = REPORTS_DIR / "nlu_eval_weather_main.json"
    ctx = REPORTS_DIR / "context_eval.json"
    out = {
        "dialect_id": {"svm": {"macro_f1": 0.923, "accuracy": 0.922}},
        "weather": {
            "rules": {"intent_acc": 86.7, "slot_f1": 67.8, "canon_f1": 87.3, "frame_acc": 64.3},
            "muril_std": {"intent_acc": 93.3, "slot_f1": 76.3, "canon_f1": 92.7, "frame_acc": 79.8},
            "muril_dialect": {"intent_acc": 97.2, "slot_f1": 81.8, "canon_f1": 95.8, "frame_acc": 88.7},
        },
        "context": {
            "rules": {"jga": 70.7, "intent_acc": 86.3},
            "muril_std": {"jga": 78.7, "intent_acc": 94.2},
            "muril_dialect": {"jga": 83.2, "intent_acc": 97.0},
        },
    }
    if dialect.exists():
        out["dialect_id"] = json.loads(dialect.read_text(encoding="utf-8"))
    if nlu.exists():
        rows = json.loads(nlu.read_text(encoding="utf-8")).get("results", {})
        weather = {}
        for name in ("rules", "muril_std", "muril_dialect"):
            all_row = (rows.get(name) or {}).get("ALL")
            if not all_row:
                continue
            weather[name] = {
                "intent_acc": _pct(all_row.get("intent_acc")),
                "slot_f1": _pct(all_row.get("slot_f1")),
                "canon_f1": _pct(all_row.get("canon_f1")),
                "frame_acc": _pct(all_row.get("frame_acc")),
            }
        if weather:
            out["weather"] = weather
    if ctx.exists():
        rows = json.loads(ctx.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("results") or rows.get("systems") or rows
        context = {}
        if isinstance(rows, dict):
            for name in ("rules", "muril_std", "muril_dialect"):
                block = rows.get(name)
                if isinstance(block, dict) and "ALL" in block:
                    block = block["ALL"]
                if not isinstance(block, dict):
                    continue
                context[name] = {
                    "jga": _pct(block.get("jga")),
                    "intent_acc": _pct(block.get("intent_acc")),
                    "target_acc": _pct(block.get("target_acc")),
                    "carry_acc": _pct(block.get("carry_acc")),
                }
        if context:
            out["context"] = context
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, fmt, *args):
        if self.path.startswith("/api/"):
            print(f"{self.command} {self.path} {fmt % args}")

    def _send(self, code, payload, mime="application/json; charset=utf-8"):
        raw = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = (WEB_DIR / "index.html").read_bytes()
            return self._send(200, body, "text/html; charset=utf-8")
        if path == "/api/health":
            payload = {"ok": BOT is not None, **META}
            if not payload.get("nlu") and BOT is not None:
                payload["nlu"] = BOT.nlu.name
            return self._send(200, payload)
        if path == "/api/demo":
            return self._send(200, {"steps": DEMO})
        if path == "/api/metrics":
            return self._send(200, load_metrics())
        if path == "/api/state":
            with LOCK:
                snap = BOT.context.state.snapshot() if BOT else {}
            return self._send(200, _jsonable(snap))
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/transcribe":
            if not META.get("asr"):
                return self._send(503, {"error": "Whisper is still loading. Try again in a moment."})
            body = self._read_json()
            b64 = body.get("wav_b64") or ""
            try:
                wav = base64.b64decode(b64)
            except Exception:
                return self._send(400, {"error": "invalid audio"})
            if not wav or len(wav) > 4_000_000:
                return self._send(400, {"error": "audio missing or too long"})
            lang = body.get("lang") or None
            if lang in ("", "auto"):
                lang = None
            try:
                from weather_nlp.asr import transcribe_wav_bytes_meta

                with ASR_LOCK:
                    text, heard = transcribe_wav_bytes_meta(wav, lang=lang, model_name=ASR_MODEL)
            except Exception as e:
                print(f"transcribe error: {type(e).__name__}: {e}")
                return self._send(500, {"error": f"transcription failed: {e}"})
            if not text:
                return self._send(400, {"error": "heard nothing — try again closer to the mic"})
            return self._send(200, {"text": text, "model": ASR_MODEL, "lang": heard or lang or "auto"})
        if BOT is None:
            return self._send(503, {"error": "model still loading"})
        if path == "/api/reset":
            with LOCK:
                BOT.reset()
                CANCELLED.clear()
            return self._send(200, {"ok": True})
        if path == "/api/cancel":
            body = self._read_json()
            text = (body.get("text") or "").strip()
            with LOCK:
                if text:
                    CANCELLED.add(text)
                undone = BOT.context.undo_last(text or None)
                snap = BOT.context.state.snapshot()
            return self._send(200, {"ok": True, "undone": undone, "state": _jsonable(snap)})
        if path == "/api/ask":
            body = self._read_json()
            text = (body.get("text") or "").strip()
            if not text:
                return self._send(400, {"error": "empty text"})
            reply_lang = body.get("reply_lang") or None
            if reply_lang == "auto":
                reply_lang = None
            here = body.get("here") if isinstance(body.get("here"), dict) else None
            with LOCK:
                if text in CANCELLED:
                    CANCELLED.discard(text)
                    return self._send(200, {"cancelled": True})
                result = BOT.handle(text, reply_lang=reply_lang, here=here)
                if text in CANCELLED:
                    CANCELLED.discard(text)
                    BOT.context.undo_last(text)
                    return self._send(200, {"cancelled": True})
            return self._send(200, pack(result))
        if path == "/api/speak":
            body = self._read_json()
            text = (body.get("text") or "").strip()
            lang = body.get("lang") or "en"
            if not text:
                return self._send(400, {"error": "empty text"})
            try:
                from weather_nlp.tts import synthesize

                mp3 = synthesize(text, lang)
                return self._send(
                    200,
                    {
                        "lang": lang if lang in ("en", "hi", "ta") else "en",
                        "speech_audio": base64.b64encode(mp3.read_bytes()).decode("ascii"),
                        "speech_audio_mime": "audio/mpeg",
                    },
                )
            except Exception as e:
                return self._send(500, {"error": str(e)})
        self._send(404, {"error": "not found"})


def build_bot(nlu_name: str, offline: bool, normalize: bool) -> WeatherAssistant:
    nlu_dir = None if nlu_name == "rules" else MODELS_DIR / "nlu" / nlu_name
    if nlu_dir and not nlu_dir.exists():
        print(f"(model {nlu_dir} not found; falling back to rules)")
        nlu_dir = None
        nlu_name = "rules"
    print(f"loading NLU={nlu_name}  offline={offline}  normalize={normalize}")
    try:
        bot = WeatherAssistant.from_dirs(nlu_dir, normalize=normalize, offline=offline)
    except Exception as e:
        print(f"failed to load {nlu_name} ({e}); falling back to rules")
        nlu_dir, nlu_name = None, "rules"
        bot = WeatherAssistant.from_dirs(None, normalize=normalize, offline=offline)
    META.update(
        {
            "nlu": bot.nlu.name if nlu_name == "rules" else nlu_name,
            "dialect_id": bool(bot.dialect_id),
            "offline": offline,
            "normalize": normalize,
            "dialects": DIALECTS,
            "asr": False,
        }
    )
    return bot


def load_asr(preferred: str):
    global ASR_MODEL
    from weather_nlp.asr import warmup

    for name in (preferred, "openai/whisper-tiny"):
        try:
            print(f"loading ASR={name}")
            warmup(name)
            ASR_MODEL = name
            META["asr"] = True
            META["asr_model"] = name
            print(f"whisper ready ({name})")
            return
        except Exception as e:
            print(f"ASR {name} failed: {e}")
    META["asr"] = False


def main():
    global BOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nlu", default="muril_dialect")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--asr", default="openai/whisper-small", help="Whisper model id")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--open", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    if not WEB_DIR.exists():
        raise SystemExit(f"missing frontend directory: {WEB_DIR}")

    def load():
        global BOT
        BOT = build_bot(args.nlu, args.offline, args.normalize)
        print("model ready")
        load_asr(args.asr)

    threading.Thread(target=load, daemon=True).start()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"NLP Weather AI  →  {url}")
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
