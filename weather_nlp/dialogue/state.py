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
        self.origin = None
        self._origin_undo = []

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
            return True
        last = self.state.turns[-1]
        self.state.intent = last.intent
        self.state.slots = copy.deepcopy(last.slots)
        self.last_weather_slots = dict.fromkeys(SLOT_KEYS)
        for turn in reversed(self.state.turns):
            if turn.intent == WEATHER_INTENT:
                self.last_weather_slots = copy.deepcopy(turn.slots)
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
            if self.carry_over and intent == WEATHER_INTENT and slots["place"] is None:
                slots["place"] = copy.deepcopy(self.last_weather_slots.get("place"))

        self.state.intent, self.state.slots = intent, slots
        if intent == WEATHER_INTENT:
            self.last_weather_slots = copy.deepcopy(slots)
        self.state.turns.append(Turn(text, lang, dialect, nlu["intent"], nlu["intent_conf"], mode, intent, copy.deepcopy(slots)))
        return self.state.snapshot(), mode
