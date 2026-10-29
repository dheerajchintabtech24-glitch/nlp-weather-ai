"""Cross-lingual elliptical follow-up benchmark.

Each item is a two-turn conversation: turn 1 is a real MASSIVE weather query in language L1; turn 2
is an elliptical follow-up ("and in Chennai?", "நாளைக்கு எப்படிங்க?") in variety L2 that changes one
slot. The gold state after turn 2 is turn 1's gold canonical state with that slot replaced. Pairing
every turn 1 with follow-ups in every variety gives a same-language control (L1 == L2, standard)
against which cross-lingual and dialectal degradation can be attributed to the language change
rather than to ellipsis.
"""

import random
import re
from collections import defaultdict

from ..config import DIALECTS, WEATHER_INTENT
from ..data.massive import load_split
from ..dialect.normalizer import _tamil_locative
from ..dialogue import Canonicalizer, ContextManager
from ..dialogue.canonicalize import SLOT_KEYS
from ..knowledge import gazetteer
from ..nlu.predictor import spans_from_tags
from ..preprocessing import fold, tokenize

DATE_FORMS = {
    ("en", "standard"): {0: "today", 1: "tomorrow", 2: "day after tomorrow"},
    ("hi", "standard"): {0: "आज", 1: "कल", 2: "परसों"},
    ("hi", "bhojpuri"): {0: "आज", 1: "काल्ह", 2: "परसों"},
    ("hi", "haryanvi"): {0: "आज", 1: "काल", 2: "परसों"},
    ("hi", "bambaiya"): {0: "आज", 1: "कल", 2: "परसों"},
    ("ta", "standard"): {0: "இன்று", 1: "நாளை", 2: "நாளை மறுநாள்"},
    ("ta", "chennai"): {0: "இன்னிக்கு", 1: "நாளைக்கு", 2: "நாளன்னைக்கு"},
    ("ta", "kongu"): {0: "இன்னைக்கு", 1: "நாளைக்கு", 2: "நாளன்னைக்கு"},
    ("ta", "madurai"): {0: "இன்னைக்கு", 1: "நாளைக்கு", 2: "நாளன்னைக்கு"},
    ("ta", "nellai"): {0: "இன்னைக்கு", 1: "நாளைக்கு", 2: "நாளன்னைக்கு"},
}

# {p} place (nominative), {pl} written locative, {pc} colloquial locative, {d} date word
TEMPLATES = {
    ("en", "standard"): {"place": ["what about {p}", "and in {p}?", "how about {p}"], "date": ["what about {d}", "and {d}?"]},
    ("hi", "standard"): {"place": ["और {p} में?", "{p} में कैसा रहेगा?", "और {p} का?"], "date": ["और {d}?", "{d} का क्या?"]},
    ("hi", "bhojpuri"): {"place": ["आ {p} में?", "{p} में का हाल बा?"], "date": ["आ {d}?", "{d} के का हाल बा?"]},
    ("hi", "haryanvi"): {"place": ["अर {p} म्ह?", "{p} म्ह के हाल सै?"], "date": ["अर {d}?", "{d} के सै?"]},
    ("hi", "bambaiya"): {"place": ["और {p} में क्या?", "{p} का क्या बोलता?"], "date": ["और {d} का क्या?", "{d} का बोल ना"]},
    ("ta", "standard"): {"place": ["{p} பற்றி என்ன?", "அப்போ {p}?", "{pl} எப்படி இருக்கும்?"], "date": ["{d} எப்படி இருக்கும்?", "அப்போ {d}?"]},
    ("ta", "chennai"): {"place": ["அப்போ {p} இன்னா?", "{pc} எப்டி?"], "date": ["{d} எப்டி?", "அப்போ {d} இன்னா?"]},
    ("ta", "kongu"): {"place": ["{pc} எப்படிங்க?", "ஏனுங்க {p}?"], "date": ["{d} எப்படிங்க?"]},
    ("ta", "madurai"): {"place": ["{pc} எப்படிண்ணே?", "அப்ப {p}?"], "date": ["{d} எப்படிண்ணே?"]},
    ("ta", "nellai"): {"place": ["{pc} எப்படிலே?", "ஏலே {p}?"], "date": ["{d} எப்படிலே?"]},
}

_TA_COLLOQUIAL = [(re.compile(p), r) for p, r in [("த்தில்$", "த்துல"), ("ையில்$", "ைல"), ("ியில்$", "ில"), ("ாவில்$", "ால"), ("([^ய])ில்$", r"\1்ல")]]


def _colloquial_locative(written: str) -> str:
    for pat, repl in _TA_COLLOQUIAL:
        if pat.search(written):
            return pat.sub(repl, written)
    return written


def varieties():
    return [(l, d) for l in ("en", "hi", "ta") for d in DIALECTS[l]]


def gold_state(rec, canon: Canonicalizer):
    nlu = {"slots": spans_from_tags(rec["tokens"], rec["tags"])}
    frame = canon.canonicalize(nlu, rec["tokens"])
    return {k: frame[k] for k in SLOT_KEYS}


def slot_key(k, v):
    if v is None:
        return None
    if k == "place":
        return v.get("canonical") or fold(v.get("surface", ""))
    if isinstance(v, dict):
        return tuple(sorted(v.items()))
    return v


def build_items(n_turn1: int = 150, seed: int = 13, langs=("en", "hi", "ta")):
    rng = random.Random(seed)
    gaz = gazetteer()
    canon = Canonicalizer(backoff=False)
    indian = [c for c in gaz.places]
    items = []
    for l1 in langs:
        pool = [r for r in load_split(l1, "test") if r["scenario"] == "weather"]
        for rec in rng.sample(pool, min(n_turn1, len(pool))):
            gold1 = gold_state(rec, canon)
            for l2, d2 in varieties():
                kind = rng.choice(["place", "date"])
                tmpl = rng.choice(TEMPLATES[(l2, d2)][kind])
                expected = dict(gold1)
                if kind == "place":
                    choices = [c for c in indian if slot_key("place", gold1["place"]) != c]
                    target = rng.choice(choices)
                    p = gaz.surface_forms(target, l2)[0]
                    pl = _tamil_locative(p) if l2 == "ta" else p
                    text = tmpl.format(p=p, pl=pl, pc=_colloquial_locative(pl))
                    expected["place"] = {"canonical": target, "surface": p}
                else:
                    cur = (gold1["date"] or {}).get("offset")
                    offset = rng.choice([o for o in (0, 1, 2) if o != cur])
                    text = tmpl.format(d=DATE_FORMS[(l2, d2)][offset])
                    expected["date"] = {"offset": offset}
                items.append({
                    "turn1": {"text": rec["text"], "tokens": rec["tokens"], "lang": l1, "gold": gold1},
                    "turn2": {"text": text, "tokens": tokenize(text), "lang": l2, "dialect": d2},
                    "changed": kind,
                    "expected": expected,
                    "condition": ("same" if l1 == l2 else "cross") + "-" + ("standard" if d2 == "standard" else "dialect"),
                })
    return items


def evaluate(items, nlu, carry_over=True, oracle_turn1=False, backoff=True, normalizer=None, dialect_id=None):
    canon = Canonicalizer(backoff=backoff)
    t1 = nlu.predict_batch([it["turn1"]["tokens"] for it in items]) if not oracle_turn1 else None
    t2_tokens = []
    for it in items:
        toks = it["turn2"]["tokens"]
        if normalizer is not None:
            d = dialect_id.predict_one(it["turn2"]["text"])[0].split(":")[1] if dialect_id else it["turn2"]["dialect"]
            toks = normalizer.normalize(toks, it["turn2"]["lang"], d)
        t2_tokens.append(toks)
    t2 = nlu.predict_batch(t2_tokens)

    stats = defaultdict(lambda: defaultdict(float))
    for i, it in enumerate(items):
        cm = ContextManager(carry_over=carry_over)
        if oracle_turn1:
            cm.state.intent = WEATHER_INTENT
            cm.state.slots = dict(it["turn1"]["gold"])
            cm.last_weather_slots = dict(it["turn1"]["gold"])
        else:
            f1 = canon.canonicalize(t1[i], it["turn1"]["tokens"])
            cm.update(it["turn1"]["text"], it["turn1"]["tokens"], it["turn1"]["lang"], t1[i], f1)
        f2 = canon.canonicalize(t2[i], t2_tokens[i])
        state, mode = cm.update(it["turn2"]["text"], it["turn2"]["tokens"], it["turn2"]["lang"], t2[i], f2, it["turn2"]["dialect"])

        exp = it["expected"]
        changed = "place" if it["changed"] == "place" else "date"
        ok_intent = state["intent"] == WEATHER_INTENT
        ok_target = slot_key(changed, state["slots"][changed]) == slot_key(changed, exp[changed])
        carried = [k for k in SLOT_KEYS if k != changed and exp[k] is not None]
        ok_carry = all(slot_key(k, state["slots"][k]) == slot_key(k, exp[k]) for k in carried)
        jga = ok_intent and all(slot_key(k, state["slots"][k]) == slot_key(k, exp[k]) for k in SLOT_KEYS)
        variety = f"{it['turn2']['lang']}:{it['turn2']['dialect']}"
        for g in ("ALL", it["condition"], f"L2={variety}", f"{it['turn1']['lang']}->{it['turn2']['lang']}"):
            s = stats[g]
            s["n"] += 1
            s["intent_acc"] += ok_intent
            s["target_acc"] += ok_target
            s["carry_acc"] += ok_carry
            s["jga"] += jga
            s["followup_detected"] += mode in ("followup", "continue")
    return {g: {k: (v if k == "n" else v / s["n"]) for k, v in s.items()} for g, s in stats.items()}
