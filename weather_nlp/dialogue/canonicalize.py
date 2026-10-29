"""Map surface slot values (any language/dialect) to language-independent canonical values."""

from ..knowledge import canonical_lexicon, gazetteer
from ..preprocessing import fold, tokenize

SLOT_KEYS = ("place", "date", "timeofday", "descriptor")
HOME = "@home"


class Canonicalizer:
    def __init__(self, backoff: bool = True):
        """``backoff``: if the NLU misses a slot, recover it by lexicon/gazetteer lookup over the utterance."""
        self.backoff = backoff
        self.lex = canonical_lexicon()
        self.gaz = gazetteer()

    def place(self, value: str):
        canon = self.gaz.lookup(value)
        if canon:
            return {"canonical": canon, "surface": value}
        if self.lex.home.find(tokenize(value)):
            return {"canonical": HOME, "surface": value}
        return {"canonical": None, "surface": value}

    def date(self, tokens, utterance_tokens):
        value = self.lex.date.first(tokens)
        if value is None:
            return None
        value = dict(value)
        # Hindi कल means both "yesterday" and "tomorrow"; tense disambiguates.
        if value.get("offset") == 1 and any(fold(t) == "कल" for t in tokens):
            if any(fold(t) in self.lex.past_markers for t in utterance_tokens):
                value["offset"] = -1
        return value

    def canonicalize(self, nlu: dict, tokens):
        frame = dict.fromkeys(SLOT_KEYS)
        sources = {}
        for s in nlu.get("slots", []):
            vt = tokenize(s["value"])
            if s["type"] == "place_name" and frame["place"] is None:
                frame["place"] = self.place(s["value"])
            elif s["type"] in ("date", "time"):
                d = self.date(vt, tokens)
                if d and frame["date"] is None:
                    frame["date"] = d
                tod = self.lex.timeofday.first(vt)
                if tod and frame["timeofday"] is None:
                    frame["timeofday"] = tod
            elif s["type"] == "timeofday" and frame["timeofday"] is None:
                frame["timeofday"] = self.lex.timeofday.first(vt)
                d = self.date(vt, tokens)
                if d and frame["date"] is None:
                    frame["date"] = d
            elif s["type"] == "weather_descriptor" and frame["descriptor"] is None:
                frame["descriptor"] = self.lex.descriptor.first(vt) or "general"
        for k, v in frame.items():
            if v is not None:
                sources[k] = "nlu"

        if self.backoff:
            if frame["place"] is None:
                spans = self.gaz.find_spans(tokens)
                if spans:
                    a, b, canon = spans[0]
                    frame["place"] = {"canonical": canon, "surface": " ".join(tokens[a:b])}
                    sources["place"] = "backoff"
            if frame["date"] is None:
                d = self.date(tokens, tokens)
                if d:
                    frame["date"], sources["date"] = d, "backoff"
            if frame["timeofday"] is None:
                tod = self.lex.timeofday.first(tokens)
                if tod:
                    frame["timeofday"], sources["timeofday"] = tod, "backoff"
            if frame["descriptor"] is None:
                de = self.lex.descriptor.first(tokens)
                if de:
                    frame["descriptor"], sources["descriptor"] = de, "backoff"
        if frame["timeofday"] and frame["date"] is None:
            frame["date"] = {"offset": 0}
        frame["_sources"] = sources
        return frame
