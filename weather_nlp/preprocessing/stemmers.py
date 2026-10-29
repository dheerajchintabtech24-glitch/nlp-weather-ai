"""Light suffix-stripping stemmers for Tamil and Hindi.

Tamil is agglutinative: a place name arrives as சென்னையில் ("in Chennai"), மதுரைக்கு ("to Madurai")
or, in spoken dialects, சென்னைல. Instead of committing to one analysis, ``tamil_stem_candidates``
returns every plausible stem so that a downstream lexicon lookup can pick the one that exists
(analysis-by-lookup), which avoids the over-stemming that a greedy stemmer suffers on Tamil.
"""

from functools import lru_cache

from .text import normalize_text

PULLI = "\u0bcd"

# (suffix, replacement). Replacement restores the stem-final form: "்" re-attaches the virama that
# disappears when a vowel-initial case marker fuses with a consonant-final stem, and "ம்" restores
# -am nouns whose oblique stem is -att- (புரம் -> புரத்தில்).
_TAMIL_RULES = [
    ("த்திலிருந்து", "ம்"),
    ("யிலிருந்து", ""),
    ("விலிருந்து", ""),
    ("ிலிருந்து", PULLI),
    ("த்திற்கு", "ம்"),
    ("த்துக்கு", "ம்"),
    ("த்தில்", "ம்"),
    ("த்தின்", "ம்"),
    ("த்துல", "ம்"),
    ("த்தோட", "ம்"),
    ("த்தை", "ம்"),
    ("களில்", ""),
    ("களை", ""),
    ("கள்", ""),
    ("விற்கு", ""),
    ("ிற்கு", PULLI),
    ("யுடன்", ""),
    ("ுடன்", PULLI),
    ("யோடு", ""),
    ("ோடு", PULLI),
    ("ுக்கு", PULLI),
    ("க்கு", ""),
    ("யில்", ""),
    ("வில்", ""),
    ("ில்", PULLI),
    ("யின்", ""),
    ("வின்", ""),
    ("ின்", PULLI),
    ("யும்", ""),
    ("வும்", ""),
    ("ும்", PULLI),
    ("யை", ""),
    ("வை", ""),
    ("்ல", PULLI),
    ("ல", ""),
    ("ை", PULLI),
    ("ா", ""),
]
_AMBIGUOUS_TAMIL = {"ை", "ா", "ல"}

_HINDI_SUFFIXES = sorted(
    """ाएंगी ाएंगे ाऊंगी ाऊंगा ाइयाँ ाइयों ाइयां
    ाएगी ाएगा ाओगी ाओगे एंगी ेंगी एंगे ेंगे ूंगी ूंगा ातीं नाओं नाएं ताओं ताएं ियाँ ियों ियां
    ाकर ाइए ाईं ाया ेगी ेगा ोगी ोगे ाने ाना ाते ाती ाता तीं ाओं ाएं ुओं ुएं ुआं
    कर ाओ िए ाई ाए ने नी ना ते ीं ती ता ाँ ां ों ें
    ो े ू ु ी ि ा""".split(),
    key=len,
    reverse=True,
)


def tamil_stem_candidates(token: str, min_stem: int = 2):
    """All analyses of ``token``, most conservative (the surface form itself) first."""
    return list(_tamil_stem_candidates(token, min_stem))


@lru_cache(maxsize=100_000)
def _tamil_stem_candidates(token, min_stem):
    token = normalize_text(token)
    out = [token]
    for suffix, repl in _TAMIL_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) >= min_stem:
            cand = token[: -len(suffix)] + repl
            if cand not in out:
                out.append(cand)
    return tuple(out)


def tamil_stem(token: str, min_stem: int = 2) -> str:
    """Single best stem: longest unambiguous suffix match."""
    token = normalize_text(token)
    for suffix, repl in _TAMIL_RULES:
        if suffix in _AMBIGUOUS_TAMIL:
            continue
        if token.endswith(suffix) and len(token) - len(suffix) >= min_stem:
            return token[: -len(suffix)] + repl
    return token


def hindi_stem(token: str, min_stem: int = 2) -> str:
    """Longest-match light stemmer after Ananthakrishnan & Rao (2003)."""
    token = normalize_text(token)
    for suffix in _HINDI_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= min_stem:
            return token[: -len(suffix)]
    return token
