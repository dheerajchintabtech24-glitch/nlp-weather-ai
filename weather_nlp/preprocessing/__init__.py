from .text import normalize_text, tokenize, fold
from .langid import detect_language, token_script
from .stemmers import tamil_stem_candidates, tamil_stem, hindi_stem

__all__ = [
    "normalize_text",
    "tokenize",
    "fold",
    "detect_language",
    "token_script",
    "tamil_stem_candidates",
    "tamil_stem",
    "hindi_stem",
]
