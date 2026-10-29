"""Unicode normalisation and tokenisation for English, Hindi (Devanagari) and Tamil."""

import re
import unicodedata
from functools import lru_cache

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_PUNCT = re.compile(r"([?!.,;:\u0964\u0965\"()\[\]{}\u2026\u201c\u201d\u2018\u2019*/\\|<>~`@#$%^&=+_-])")
_SPACES = re.compile(r"\s+")

NUKTA = "\u093c"
CHANDRABINDU = "\u0901"
ANUSVARA = "\u0902"


def normalize_text(text: str) -> str:
    """NFC-normalise, strip zero-width joiners, lower-case Latin and collapse whitespace."""
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return _SPACES.sub(" ", text).strip().lower()


def tokenize(text: str):
    """Whitespace tokenisation after detaching punctuation; punctuation tokens are dropped.

    Apostrophes are kept inside tokens so English forms like "what's" survive, matching MASSIVE.
    """
    text = normalize_text(text)
    text = _PUNCT.sub(r" \1 ", text)
    return [t for t in text.split() if not _PUNCT.fullmatch(t) and t != "'"]


@lru_cache(maxsize=200_000)
def fold(token: str) -> str:
    """Aggressive normalisation used only for dictionary lookup.

    Removes Devanagari nukta (ज़ -> ज) and maps chandrabindu to anusvara (हूँ -> हूं), two common
    sources of spelling variation in user-typed Hindi.
    """
    token = unicodedata.normalize("NFD", normalize_text(token))
    token = token.replace(NUKTA, "").replace(CHANDRABINDU, ANUSVARA)
    return unicodedata.normalize("NFC", token)
