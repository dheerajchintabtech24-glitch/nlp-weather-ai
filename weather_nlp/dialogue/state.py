"""Language-independent conversation state with cross-lingual slot carry-over.

The state holds only symbols: a shared intent label and canonical slot values (place = canonical
English key, date = day offset / range, descriptor = attribute code). Nothing in it depends on the
language or dialect of the turn that produced it, so an elliptical Tamil follow-up can inherit what
a Hindi turn established.
"""

import copy
from dataclasses import dataclass, field

from ..config import WEATHER_INTENT
from ..knowledge import canonical_lexicon
from .canonicalize import SLOT_KEYS


@dataclass
class Turn:
    text: str
    lang: str
    dialect: str
    nlu_intent: str
    nlu_conf: float
    mode: str
    intent: str
    slots: dict


@dataclass
class DialogueState:
    intent: str = None
    slots: dict = field(default_factory=lambda: dict.fromkeys(SLOT_KEYS))
    turns: list = field(default_factory=list)

    def snapshot(self):
        return {"intent": self.intent, "slots": copy.deepcopy(self.slots)}


class ContextManager:
    def __init__(self, followup_conf: float = 0.7, short_len: int = 4, short_conf: float = 0.9, carry_over: bool = True):
        """
        followup_conf: below this intent confidence, a turn that carries slots is read as a follow-up.
        short_len / short_conf: turns of at most ``short_len`` tokens that carry slots are elliptical
            unless the NLU is at least ``short_conf`` sure of a different intent ("remind me tomorrow").
        carry_over: disable to get the no-context baseline used in evaluation.
        """
        self.followup_conf = followup_conf
        self.short_len = short_len
        self.short_conf = short_conf
        self.carry_over = carry_over
        self.lex = canonical_lexicon()
        self.reset()

    def reset(self):
        self.state = DialogueState()
        self.last_weather_slots = dict.fromkeys(SLOT_KEYS)
        self.resolved_place = None
        self.origin = None
        self._origin_undo = []

    @staticmethod
    def _resolved(place) -> bool:
        """A place the gazetteer or geocoder actually named. A bare surface is not enough."""
        return bool(place) and bool(place.get("canonical"))

    def undo_last(self, text=None) -> bool:
        """Drop the last turn if it matches `text` (or is the only candidate) and restore slots."""
        if not self.state.turns:
            return False
        if text is not None and self.state.turns[-1].text != text:
            return False
        self.state.turns.pop()
        prev_origin = self._origin_undo.pop() if self._origin_undo else None
        self.origin = prev_origin
        if not self.state.turns:
            self.state = DialogueState()
            self.last_weather_slots = dict.fromkeys(SLOT_KEYS)
            self.resolved_place = None
            return True
        last = self.state.turns[-1]
        self.state.intent = last.intent
        self.state.slots = copy.deepcopy(last.slots)
        self.last_weather_slots = dict.fromkeys(SLOT_KEYS)
        self.resolved_place = None
        for turn in reversed(self.state.turns):
            if turn.intent == WEATHER_INTENT:
                self.last_weather_slots = copy.deepcopy(turn.slots)
                if self._resolved(turn.slots.get("place")):
                    self.resolved_place = copy.deepcopy(turn.slots["place"])
                break
        return True

    def _has_marker(self, tokens):
        return any(a <= 1 for a, _, _, _ in self.lex.followup.find(tokens)) or any(
            t.startswith("பற்றி") for t in tokens
        )

    def update(self, text, tokens, lang, nlu, frame, dialect="standard"):
        new = {k: frame.get(k) for k in SLOT_KEYS if frame.get(k) is not None}
        prev = self.state.intent if self.carry_over else None
        marker = self._has_marker(tokens)

        if prev and nlu["intent"] == prev:
            mode = "continue"
        elif prev and new and (
            marker
            or nlu["intent_conf"] < self.followup_conf
            or (len(tokens) <= self.short_len and nlu["intent_conf"] < self.short_conf)
        ):
            mode = "followup"
        else:
            mode = "new"

        if mode in ("continue", "followup"):
            intent = prev
            slots = copy.deepcopy(self.state.slots)
            slots.update(new)
        else:
            intent = nlu["intent"]
            slots = dict.fromkeys(SLOT_KEYS)
            slots.update(new)
            if self.carry_over and intent == WEATHER_INTENT and slots.get("place") is None:
                fallback = self.resolved_place if self._resolved(self.resolved_place) else self.last_weather_slots.get("place")
                if self._resolved(fallback):
                    slots["place"] = copy.deepcopy(fallback)

        self.state.intent, self.state.slots = intent, slots
        if intent == WEATHER_INTENT:
            self.last_weather_slots = copy.deepcopy(slots)
        self.state.turns.append(Turn(text, lang, dialect, nlu["intent"], nlu["intent_conf"], mode, intent, copy.deepcopy(slots)))
        return self.state.snapshot(), mode

    def remember_place(self, place: dict):
        """Remember a city the weather service actually resolved, so 'there' follows that city."""
        if not self._resolved(place):
            return
        rec = {
            "canonical": place.get("canonical"),
            "surface": place.get("surface") or place.get("canonical"),
            "lat": place.get("lat"),
            "lon": place.get("lon"),
            "names": copy.deepcopy(place.get("names")),
        }
        self.resolved_place = rec
        self.state.slots["place"] = copy.deepcopy(rec)
        self.last_weather_slots["place"] = copy.deepcopy(rec)
        if self.state.turns:
            self.state.turns[-1].slots["place"] = copy.deepcopy(rec)

    def drop_failed_place(self):
        """A place the weather service could not find must not be what the next 'there' means."""
        keep = copy.deepcopy(self.resolved_place) if self._resolved(self.resolved_place) else None
        self.state.slots["place"] = copy.deepcopy(keep)
        self.last_weather_slots["place"] = copy.deepcopy(keep)
        if self.state.turns:
            self.state.turns[-1].slots["place"] = copy.deepcopy(keep)
