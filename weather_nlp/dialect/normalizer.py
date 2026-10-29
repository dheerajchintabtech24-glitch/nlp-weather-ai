"""Dialect -> standard normalisation, conditioned on the identified dialect.

Conditioning matters because dialect markers collide with standard words: Bhojpuri का ("what")
is the standard Hindi genitive का, and Haryanvi के ("what") is the genitive के. Applying every
dialect's inverse map unconditionally would corrupt standard input, so the normaliser only uses
the inverse map of the dialect predicted by the dialect identifier.
"""

from ..knowledge import gazetteer
from ..preprocessing import fold, tamil_stem_candidates, token_script
from .lexicon import DialectLexicon

PULLI = "\u0bcd"
_TA_VOWEL_SIGNS_Y = set("ிீெேைை")


def _tamil_locative(stem: str) -> str:
    """Standard written locative for a Tamil noun stem (சென்னை -> சென்னையில்)."""
    if stem.endswith("ம்"):
        return stem[:-2] + "த்தில்"
    if stem.endswith(PULLI):
        return stem[:-1] + "ில்"
    if stem[-1] in _TA_VOWEL_SIGNS_Y:
        return stem + "யில்"
    return stem + "வில்"


class LexiconNormalizer:
    def __init__(self, lexicon: DialectLexicon = None):
        self.lexicon = lexicon or DialectLexicon()
        self._inverse = {}

    def _build(self, lang, dialect):
        layer = self.lexicon.layer(lang, dialect)
        phrases, words = {}, {}

        def add(std_tokens, variant):
            key = tuple(fold(t) for t in variant)
            if tuple(fold(t) for t in std_tokens) == key:
                return
            target = phrases if len(variant) > 1 else words
            target.setdefault(key if len(variant) > 1 else key[0], list(std_tokens))

        for std, variants in layer.phrases.items():
            for v in variants:
                add(list(std), v)
        for std, variants in layer.words.items():
            for v in variants:
                add([std], v)
        prefixes = sorted(({tuple(fold(t) for t in p) for p in layer.prefix_particles}), key=len, reverse=True)
        suffixes = sorted(({tuple(fold(t) for t in p) for p in layer.suffix_particles}), key=len, reverse=True)
        return {
            "phrases": phrases,
            "words": words,
            "max_len": max((len(k) for k in phrases), default=1),
            "prefixes": prefixes,
            "suffixes": suffixes,
            "attach": [fold(a) for a in layer.attach_particles],
        }

    def inverse(self, lang, dialect):
        key = (lang, dialect)
        if key not in self._inverse:
            self._inverse[key] = self._build(lang, dialect)
        return self._inverse[key]

    def normalize(self, tokens, lang: str, dialect: str):
        if dialect in (None, "standard") or dialect not in self.lexicon.dialects(lang):
            return list(tokens)
        inv = self.inverse(lang, dialect)
        toks = [fold(t) for t in tokens]

        for p in inv["prefixes"]:
            if tuple(toks[: len(p)]) == p and len(toks) > len(p):
                toks = toks[len(p) :]
                break
        for s in inv["suffixes"]:
            if len(toks) > len(s) and tuple(toks[-len(s) :]) == s:
                toks = toks[: -len(s)]
                break
        if toks:
            for a in inv["attach"]:
                if toks[-1].endswith(a) and len(toks[-1]) > len(a) + 1:
                    toks[-1] = toks[-1][: -len(a)]
                    break

        out, i = [], 0
        while i < len(toks):
            for L in range(min(inv["max_len"], len(toks) - i), 1, -1):
                std = inv["phrases"].get(tuple(toks[i : i + L]))
                if std:
                    out += std
                    i += L
                    break
            else:
                tok = toks[i]
                if tok in inv["words"]:
                    out += inv["words"][tok]
                elif lang == "ta" and token_script(tok) == "ta" and tok.endswith("ல"):
                    out.append(self._restore_tamil_locative(tok))
                else:
                    out.append(tok)
                i += 1
        return out

    @staticmethod
    def _restore_tamil_locative(tok: str) -> str:
        # Only rewrite colloquial -ல when the stem is a known place, so ordinary words ending in ல
        # (நல்ல, சொல்ல) are left alone.
        gaz = gazetteer()
        for cand in tamil_stem_candidates(tok)[1:]:
            if fold(cand) in gaz.index:
                return _tamil_locative(cand)
        return tok
