"""Spoken form of a weather reply: number words, then optional dialect wording.

The voice is always the question language (en / hi / ta). Dialect wording reuses the
existing standard-to-dialect lexicons. There is no Kongu, Madurai, Nellai, Chennai or
Bambaiya TTS voice, so those replies keep the Hindi or Tamil voice and only change words.
Discourse particles are left off: they mark questions, not forecasts.
"""

import re

from .dialect.lexicon import DialectLexicon
from .preprocessing.text import fold, normalize_text

# Applied only when the dialect classifier is confident. Below this, speak standard wording.
DIALECT_MIN_CONF = 0.45

_EN_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_HI = [
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सतासी", "अठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पंचानवे", "छियानवे", "सतानवे", "अट्ठानवे", "निन्यानवे",
    "सौ",
]

_TA_ONES = [
    "பூஜ்ஜியம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது",
    "பத்து", "பதினொன்று", "பன்னிரண்டு", "பதிமூன்று", "பதினான்கு", "பதினைந்து", "பதினாறு", "பதினேழு", "பதினெட்டு", "பத்தொன்பது",
]
_TA_TENS = {
    20: "இருபது", 30: "முப்பது", 40: "நாற்பது", 50: "ஐம்பது",
    60: "அறுபது", 70: "எழுபது", 80: "எண்பது", 90: "தொண்ணூறு",
}
_TA_STEMS = {
    2: "இருபத்தி", 3: "முப்பத்தி", 4: "நாற்பத்தி", 5: "ஐம்பத்தி",
    6: "அறுபத்தி", 7: "எழுபத்தி", 8: "எண்பத்தி", 9: "தொண்ணூற்றி",
}

_UNITS = {
    "en": {"deg": "degrees Celsius", "pct": "percent", "wind": "kilometres per hour", "to": "to", "minus": "minus"},
    "hi": {"deg": "डिग्री सेल्सियस", "pct": "प्रतिशत", "wind": "किलोमीटर प्रति घंटा", "to": "से", "minus": "माइनस"},
    "ta": {"deg": "டிகிரி செல்சியஸ்", "pct": "சதவீதம்", "wind": "கிலோமீட்டர் வேகம்", "to": "முதல்", "minus": "மைனஸ்"},
}

_PAUSE = set(".,;:?!।")
_SPLIT = re.compile(r"([?!.,;:\u0964])")
_DEG_RANGE = re.compile(r"(?<!\d)(-?\d+)\s*[–—-]\s*(-?\d+)\s*°\s*C\b", re.I)
_DEG = re.compile(r"(?<!\d)(-?\d+)\s*°\s*C\b", re.I)
_PCT = re.compile(r"(?<!\d)(-?\d+)\s*%")
_WIND = re.compile(r"(?<!\d)(-?\d+)\s*(?:km/h|किमी/घंटा|கி\.மீ)")
_RANGE = re.compile(r"(?<!\d)(-?\d+)\s*[–—]\s*(-?\d+)(?!\d)")
_INT = re.compile(r"(?<!\d)(-?\d+)(?!\d)")


def _int_word(n: int, lang: str) -> str:
    u = _UNITS[lang]
    if n < 0:
        return u["minus"] + " " + _int_word(-n, lang)
    if lang == "en":
        if n < 20:
            return _EN_ONES[n]
        if n < 100:
            tens, ones = divmod(n, 10)
            return _EN_TENS[tens] if ones == 0 else f"{_EN_TENS[tens]}-{_EN_ONES[ones]}"
        if n == 100:
            return "one hundred"
        return str(n)
    if lang == "hi":
        return _HI[n] if 0 <= n < len(_HI) else str(n)
    if n < 20:
        return _TA_ONES[n]
    if n == 100:
        return "நூறு"
    if n < 100 and n % 10 == 0:
        return _TA_TENS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return f"{_TA_STEMS[tens]} {_TA_ONES[ones]}"
    return str(n)


def to_speech_text(text: str, lang: str) -> str:
    """Expand temperatures, percentages, wind and leftover integers into words."""
    lang = lang if lang in _UNITS else "en"
    u = _UNITS[lang]
    text = text.replace("\n", ". ")
    text = _DEG_RANGE.sub(lambda m: f"{_int_word(int(m.group(1)), lang)} {u['to']} {_int_word(int(m.group(2)), lang)} {u['deg']}", text)
    text = _DEG.sub(lambda m: f"{_int_word(int(m.group(1)), lang)} {u['deg']}", text)
    text = _PCT.sub(lambda m: f"{_int_word(int(m.group(1)), lang)} {u['pct']}", text)
    text = _WIND.sub(lambda m: f"{_int_word(int(m.group(1)), lang)} {u['wind']}", text)
    text = _RANGE.sub(lambda m: f"{_int_word(int(m.group(1)), lang)} {u['to']} {_int_word(int(m.group(2)), lang)}", text)
    text = _INT.sub(lambda m: _int_word(int(m.group(1)), lang), text)
    return re.sub(r"\s+", " ", text).strip()


def _pick(variants, std_tokens):
    std_key = tuple(fold(t) for t in std_tokens)
    for variant in variants:
        if tuple(fold(t) for t in variant) != std_key:
            return variant
    return None


def _pieces(text: str):
    text = normalize_text(text)
    text = _SPLIT.sub(r" \1 ", text)
    return [t for t in text.split() if t]


def _join(pieces):
    out = []
    for piece in pieces:
        if piece in _PAUSE:
            if out:
                out[-1] += piece
            continue
        out.append(piece)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def dialectize(text: str, lang: str, dialect: str, lexicon: DialectLexicon = None) -> str:
    """Replace standard words with the first differing dialect variant. No particles."""
    if lang not in ("hi", "ta") or not dialect or dialect == "standard":
        return text
    lexicon = lexicon or DialectLexicon()
    if dialect not in lexicon.raw.get(lang, {}).get("dialects", {}):
        return text
    layer = lexicon.layer(lang, dialect)
    pieces = _pieces(text)
    folded = [fold(p) if p not in _PAUSE else p for p in pieces]
    phrases = sorted(layer.phrases.items(), key=lambda kv: len(kv[0]), reverse=True)
    out = []
    i = 0
    while i < len(pieces):
        if pieces[i] in _PAUSE:
            out.append(pieces[i])
            i += 1
            continue
        matched = False
        for key, variants in phrases:
            n = len(key)
            if n < 2 or i + n > len(pieces):
                continue
            if any(pieces[i + j] in _PAUSE for j in range(n)):
                continue
            if tuple(folded[i : i + n]) != key:
                continue
            pick = _pick(variants, pieces[i : i + n])
            if not pick:
                continue
            out.extend(pick)
            i += n
            matched = True
            break
        if matched:
            continue
        variants = layer.words.get(folded[i])
        pick = _pick(variants, [pieces[i]]) if variants else None
        if pick:
            out.extend(pick)
        else:
            out.append(pieces[i])
        i += 1
    return _join(out)


def prepare_speech(text: str, lang: str, dialect: str = "standard", dialect_conf: float = 0.0, lexicon: DialectLexicon = None) -> dict:
    """Number expansion in the question language, then dialect wording when the ID is confident."""
    lang = lang if lang in _UNITS else "en"
    base = to_speech_text(text, lang)
    use = (
        dialect
        if lang in ("hi", "ta") and dialect not in (None, "", "standard") and dialect_conf >= DIALECT_MIN_CONF
        else "standard"
    )
    spoken = dialectize(base, lang, use, lexicon) if use != "standard" else base
    applied = use if spoken != base else "standard"
    return {
        "speech_lang": lang,
        "speech_base": base,
        "speech_text": spoken,
        "speech_dialect": applied,
    }
