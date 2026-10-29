"""Language identification: native script first, then romanized Hindi / Tamil cues."""

from collections import Counter
import re

_RANGES = {
    "hi": (0x0900, 0x097F),
    "ta": (0x0B80, 0x0BFF),
}

# Latin-script weather / everyday cues when Whisper or the user types romanized Indic.
_ROMAN_HI = {
    "mausam", "mosam", "mausum", "baarish", "barish", "barkha", "barsaat", "barsat",
    "garmi", "sardi", "thandi", "thanda", "garam", "aaj", "kal", "parso", "parson",
    "kaisa", "kesa", "kaisi", "kesi", "kaise", "kese", "kya", "kyun", "kyu",
    "hai", "hain", "ho", "hoga", "hogi", "mein", "me", "mujhe", "batao", "bataao",
    "dilli", "delhi", "mumbai", "bengaluru", "bangalore", "chennai", "kolkata",
    "nahi", "nahin", "bahut", "abhi", "yahan", "wahan", "kitna", "kitni",
    "hindustan", "bharat", "rooh", "hawa", "dhup", "dhoop", "andhera",
}
_ROMAN_TA = {
    "mazhai", "malai", "vanilai", "vela", "epdi", "eppadi", "enna", "irukku", "irukka",
    "naalaiku", "naalai", "inniku", "innikki", "ippo", "romba", "konjam",
    "chennai", "kovai", "madurai", "nellai", "coimbatore", "bangalore", "bengaluru",
    "sollu", "theriyum", "illai", "vendum", "venum",
}

_WORD = re.compile(r"[a-zA-Z']+")


def char_script(ch: str):
    cp = ord(ch)
    for lang, (lo, hi) in _RANGES.items():
        if lo <= cp <= hi:
            return lang
    if ch.isascii() and ch.isalpha():
        return "en"
    return None


def token_script(token: str):
    counts = Counter(s for s in map(char_script, token) if s)
    return counts.most_common(1)[0][0] if counts else None


def _roman_votes(text: str):
    words = {w.lower() for w in _WORD.findall(text)}
    if not words:
        return {}
    return {
        "hi": len(words & _ROMAN_HI) / len(words),
        "ta": len(words & _ROMAN_TA) / len(words),
    }


def detect_language(text: str, default: str = "en"):
    """Return (language, per-language share).

    Native scripts win. If the utterance is mostly Latin, romanized Hindi/Tamil
    keyword density can override English (e.g. "Delli mein mosam kesa hai").
    """
    counts = Counter(s for s in map(char_script, text) if s)
    total = sum(counts.values())
    if not total:
        return default, {}
    shares = {k: v / total for k, v in counts.items()}
    script_lang = max(shares, key=shares.get)

    if script_lang != "en" or shares.get("en", 0) < 0.85:
        return script_lang, shares

    roman = _roman_votes(text)
    hi_v, ta_v = roman.get("hi", 0.0), roman.get("ta", 0.0)
    # Need a clear cue density so plain English weather questions stay English.
    if hi_v >= 0.18 and hi_v >= ta_v and hi_v > ta_v + 0.02:
        shares = {"hi": hi_v, "en": 1.0 - hi_v}
        return "hi", shares
    if ta_v >= 0.18 and ta_v > hi_v:
        shares = {"ta": ta_v, "en": 1.0 - ta_v}
        return "ta", shares
    if hi_v >= 0.12 and sum(1 for w in _WORD.findall(text) if w.lower() in _ROMAN_HI) >= 2:
        shares = {"hi": hi_v, "en": max(0.0, 1.0 - hi_v)}
        return "hi", shares
    return script_lang, shares
