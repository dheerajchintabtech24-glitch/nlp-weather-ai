"""End-to-end pipeline: text -> language/dialect ID -> NLU -> canonical state -> tool -> response.

Dialect->standard normalisation is optional and off by default. With muril_dialect (trained on
dialect forms) it does not help; keep --normalize only as an ablation / for muril_std.
"""

import copy
from pathlib import Path

from .config import DEFAULT_LOCATION, MODELS_DIR, WEATHER_INTENT
from .dialect.identifier import DialectIdentifier
from .dialogue import Canonicalizer, ContextManager
from .dialogue.state import Turn
from .nlg import Responder
from .nlu.predictor import RuleNLU, TransformerNLU
from .preprocessing import detect_language, tokenize
from .speech_text import prepare_speech
from .tools import ToolRouter, WeatherTool
from .tools.travel import classify, estimate
from .tools.weather import WeatherError


class WeatherAssistant:
    def __init__(
        self,
        nlu=None,
        dialect_id: DialectIdentifier = None,
        normalize: bool = False,
        offline: bool = False,
        backoff: bool = True,
        carry_over: bool = True,
        reply_lang: str = None,
        weather: WeatherTool = None,
    ):
        self.nlu = nlu or RuleNLU()
        self.dialect_id = dialect_id
        self.normalize = normalize
        self.normalizer = None
        if normalize:
            from .dialect import LexiconNormalizer

            self.normalizer = LexiconNormalizer()
        self.canon = Canonicalizer(backoff=backoff)
        self.context = ContextManager(carry_over=carry_over)
        self.weather = weather or WeatherTool(offline=offline)
        self.router = ToolRouter(self.weather)
        self.responder = Responder()
        self.reply_lang = reply_lang

    @classmethod
    def from_dirs(cls, nlu_dir: Path = None, dialect_model: Path = None, **kw):
        nlu = TransformerNLU(nlu_dir) if nlu_dir and Path(nlu_dir).exists() else RuleNLU()
        dialect_model = Path(dialect_model) if dialect_model else MODELS_DIR / "dialect_id" / "svm.joblib"
        dialect_id = DialectIdentifier.load(dialect_model) if dialect_model.exists() else None
        return cls(nlu=nlu, dialect_id=dialect_id, **kw)

    def reset(self):
        self.context.reset()

    def identify_dialect(self, text, lang, nlu=None):
        if self.dialect_id is not None:
            label, conf = self.dialect_id.predict_one(text)
            return label.split(":")[1], conf, "classifier"
        if nlu and nlu.get("dialect"):
            return nlu["dialect"].split(":")[1], nlu.get("dialect_conf", 0.0), "nlu_head"
        return "standard", 0.0, "none"

    def dialect_distribution(self, text, lang):
        """Per-dialect scores, restricted to the script-detected language and re-normalised."""
        if self.dialect_id is None:
            return []
        labels = self.dialect_id.labels
        row = self.dialect_id.scores([text])[0]
        items = []
        for lab, score in zip(labels, row):
            l, d = lab.split(":", 1)
            if l == lang:
                items.append({"dialect": d, "score": float(score)})
        total = sum(i["score"] for i in items) or 1.0
        for i in items:
            i["score"] = i["score"] / total
        items.sort(key=lambda x: -x["score"])
        return items

    def _locate(self, slot: dict) -> dict:
        if slot.get("lat") is not None and slot.get("lon") is not None:
            name = slot.get("canonical") or "your location"
            names = slot.get("names") or {"en": name, "hi": "आपकी जगह", "ta": "உங்கள் இடம்"}
            return {"canonical": name, "lat": slot["lat"], "lon": slot["lon"], "names": names, "surface": slot.get("surface") or name}
        return self.weather.resolve_place(slot)

    def _here(self, here) -> dict:
        if not here or here.get("lat") is None or here.get("lon") is None:
            return None
        return {
            "canonical": "your location",
            "surface": "your location",
            "lat": float(here["lat"]),
            "lon": float(here["lon"]),
            "names": {"en": "your location", "hi": "आपकी जगह", "ta": "உங்கள் இடம்"},
        }

    def _trace(self, lang, shares, dialect, dconf, dsrc, nlu, frame, mode, state, result, note) -> list:
        share = ", ".join(f"{k} {v:.0%}" for k, v in sorted(shares.items(), key=lambda kv: -kv[1])) or "no script letters"
        place = (state.get("slots") or {}).get("place") or {}
        fresh = (frame or {}).get("place") or {}
        carried = place.get("canonical") and fresh.get("canonical") != place.get("canonical")
        place_bit = place.get("canonical") or place.get("surface") or "none"
        inherit = "carried from the previous turn, whatever language that turn used" if carried or mode in ("followup", "continue") else "read from this turn"
        intent = nlu.get("intent")
        conf = float(nlu.get("intent_conf") or 0)
        slots = ", ".join(s.get("type", "") for s in nlu.get("slots") or []) or "none"
        tool = result.get("tool") or result.get("status")
        detail = note or tool
        if result.get("status") == "ok" and (result.get("payload") or {}).get("kind") == "travel":
            trip = result["payload"]["trip"]
            detail = (
                f"{trip['source']} route {trip['km']} km, clear {trip['base_minutes']} min, "
                f"weather ×{trip['factor']} ({trip['reason']}) → {trip['minutes']} min"
            )
        return [
            {"title": "Language identification", "detail": f"{lang}. Script shares: {share}."},
            {"title": "Dialect classification", "detail": f"{dialect} at {dconf:.0%} ({dsrc}). Character n-grams, script-gated."},
            {"title": "Intent and slots", "detail": f"{intent} at {conf:.0%}. Slots: {slots}."},
            {"title": "Canonical form", "detail": f"Place {place_bit}. Date and descriptor are language-independent symbols."},
            {"title": "Dialogue state", "detail": f"Mode {mode}. Place {inherit}."},
            {"title": "Grounding", "detail": detail},
            {"title": "Reply", "detail": "Template in the question language. Every number comes from the tool payload, then a neural voice."},
        ]

    def _pack(self, text, tokens, lang, shares, dialect, dconf, dsrc, nlu, frame, mode, state, result, reply_lang, trace):
        out_lang = reply_lang or self.reply_lang or lang
        speech_lang = lang if lang in ("en", "hi", "ta") else "en"
        written = self.responder.render(result, out_lang)
        spoken_src = written if out_lang == speech_lang else self.responder.render(result, speech_lang)
        speech = prepare_speech(spoken_src, speech_lang, dialect, float(dconf or 0.0))
        nlu_public = {k: nlu[k] for k in ("intent", "intent_conf", "slots") if k in nlu}
        if "tags" in nlu:
            nlu_public["tags"] = nlu["tags"]
        return {
            "text": text,
            "tokens": tokens,
            "lang": lang,
            "script_shares": shares,
            "dialect": dialect,
            "dialect_conf": float(dconf or 0.0),
            "dialect_source": dsrc,
            "dialect_scores": self.dialect_distribution(text, lang),
            "normalized": tokens if self.normalize else None,
            "nlu": nlu_public,
            "frame": frame,
            "mode": mode,
            "state": state,
            "tool": result,
            "trace": trace,
            "reply_lang": out_lang,
            "response": written,
            "speech_lang": speech["speech_lang"],
            "speech_base": speech["speech_base"],
            "speech_text": speech["speech_text"],
            "speech_dialect": speech["speech_dialect"],
        }

    def handle(self, text: str, reply_lang: str = None, here=None) -> dict:
        tokens = tokenize(text)
        lang, shares = detect_language(text)
        dialect, dconf, dsrc = self.identify_dialect(text, lang)
        nlu_tokens = self.normalizer.normalize(tokens, lang, dialect) if self.normalize else tokens
        if nlu_tokens:
            nlu = self.nlu.predict(nlu_tokens)
        else:
            nlu = {"intent": "other", "intent_conf": 0.0, "slots": [], "tags": []}
        # Romanized Hindi/Tamil often fools the transformer (MASSIVE is native script). Fall back
        # to the lexicon NLU when it clearly sees a weather cue the neural model missed.
        if nlu.get("intent") != WEATHER_INTENT and nlu_tokens and not isinstance(self.nlu, RuleNLU):
            rule = RuleNLU().predict(nlu_tokens)
            if rule.get("intent") == WEATHER_INTENT:
                nlu = rule
        if dsrc == "none" and nlu.get("dialect"):
            dialect, dconf, dsrc = self.identify_dialect(text, lang, nlu)
        frame = self.canon.canonicalize(nlu, nlu_tokens)
        kind, origin_mention, dest_mention = classify(text)
        before_origin = self.context.origin

        if kind == "set_origin":
            try:
                placed = self._locate(origin_mention)
            except WeatherError as e:
                result = {"status": "error", "error": e.kind, "detail": e.detail}
                state = self.context.state.snapshot()
                trace = self._trace(lang, shares, dialect, dconf, dsrc, nlu, frame, "origin", state, result, None)
                return self._pack(text, tokens, lang, shares, dialect, dconf, dsrc, nlu, frame, "origin", state, result, reply_lang, trace)
            self.context.origin = {"canonical": placed["canonical"], "surface": origin_mention.get("surface") or placed["canonical"]}
            self.context._origin_undo.append(before_origin)
            kept = self.context.state.intent or WEATHER_INTENT
            self.context.state.turns.append(
                Turn(text, lang, dialect, "origin_set", 1.0, "origin", kept, copy.deepcopy(self.context.state.slots))
            )
            result = {"status": "ok", "tool": "origin", "payload": {"kind": "origin", "place": placed}}
            state = self.context.state.snapshot()
            shown = {**nlu, "intent": "origin_set", "intent_conf": 1.0}
            trace = self._trace(lang, shares, dialect, dconf, dsrc, shown, frame, "origin", state, result, f"Start city set to {placed['canonical']}. Weather slots were left as they were.")
            return self._pack(text, tokens, lang, shares, dialect, dconf, dsrc, shown, frame, "origin", state, result, reply_lang, trace)

        if kind == "trip":
            frame = dict(frame)
            # No city in this sentence ("how long to go there?") must not keep a failed lookup.
            frame["place"] = dest_mention if dest_mention else None
            nlu = {**nlu, "intent": WEATHER_INTENT, "intent_conf": max(float(nlu.get("intent_conf") or 0), 0.8)}

        state, mode = self.context.update(text, tokens, lang, nlu, frame, dialect)
        self.context._origin_undo.append(before_origin)

        if kind == "trip":
            if origin_mention:
                self.context.origin = origin_mention
            carried = (state.get("slots") or {}).get("place")
            if not ContextManager._resolved(carried):
                carried = self.context.resolved_place
            dest_slot = dest_mention or carried
            origin_slot = origin_mention or self.context.origin or self._here(here) or {"canonical": DEFAULT_LOCATION, "surface": DEFAULT_LOCATION}
            if not dest_slot or not (dest_slot.get("canonical") or dest_slot.get("surface")):
                result = {"status": "error", "error": "need_place", "detail": ""}
            else:
                try:
                    origin = self._locate(origin_slot)
                    forecast = self.weather.query({**state["slots"], "place": dest_slot})
                    dest = forecast["place"]
                    if origin.get("canonical") and origin["canonical"] == dest.get("canonical"):
                        result = {"status": "ok", "tool": "travel", "payload": {"kind": "same_place", "place": dest, "days": forecast.get("days") or []}}
                    else:
                        day = forecast["days"][0]
                        trip = estimate(origin, dest, day, offline=self.weather.offline)
                        result = {"status": "ok", "tool": "travel", "payload": {**forecast, "kind": "travel", "origin": origin, "trip": trip}}
                    self.context.remember_place(dest)
                    state = self.context.state.snapshot()
                except WeatherError as e:
                    result = {"status": "error", "error": e.kind, "detail": e.detail}
                    if e.kind == "place_not_found":
                        self.context.drop_failed_place()
                        state = self.context.state.snapshot()
            shown = {**nlu, "intent": "travel_time", "intent_conf": 0.9}
            trace = self._trace(lang, shares, dialect, dconf, dsrc, shown, frame, mode, state, result, None)
            return self._pack(text, tokens, lang, shares, dialect, dconf, dsrc, shown, frame, mode, state, result, reply_lang, trace)

        result = self.router.route(state) if state["intent"] == WEATHER_INTENT else {"status": "unsupported"}
        if result.get("status") == "ok" and (result.get("payload") or {}).get("place"):
            self.context.remember_place(result["payload"]["place"])
            state = self.context.state.snapshot()
        elif result.get("error") == "place_not_found":
            self.context.drop_failed_place()
            state = self.context.state.snapshot()
        note = "Open-Meteo forecast for the canonical place." if result.get("status") == "ok" else result.get("error") or result.get("status")
        trace = self._trace(lang, shares, dialect, dconf, dsrc, nlu, frame, mode, state, result, note)
        return self._pack(text, tokens, lang, shares, dialect, dconf, dsrc, nlu, frame, mode, state, result, reply_lang, trace)
