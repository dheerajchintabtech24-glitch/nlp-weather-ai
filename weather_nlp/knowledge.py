"""Language-independent knowledge: place gazetteer and canonical value lexicons."""

import json
from functools import lru_cache

from .config import RESOURCES
from .preprocessing import fold, tamil_stem_candidates, token_script, tokenize


class FormMatcher:
    """Finds non-overlapping, longest-first occurrences of multi-lingual surface forms.

    Tamil forms match as token prefixes so that inflected forms (நாளைக்கு, வெயிலாக) hit the base
    entry; other forms must match exactly after folding.
    """

    def __init__(self, entries):
        # entries: list of (form_tokens, value, lang)
        entries = sorted(entries, key=lambda e: (-len(e[0]), -len("".join(e[0]))))
        self.entries = [
            (tuple(fold(t) for t in form), tuple(token_script(t) == "ta" for t in form), value, lang)
            for form, value, lang in entries
        ]

    def find(self, tokens):
        tokens = [fold(t) for t in tokens]
        used = [False] * len(tokens)
        hits = []
        for form, prefix, value, lang in self.entries:
            L = len(form)
            for i in range(len(tokens) - L + 1):
                if any(used[i : i + L]):
                    continue
                if all(tokens[i + j].startswith(form[j]) if prefix[j] else tokens[i + j] == form[j] for j in range(L)):
                    hits.append((i, i + L, value, lang))
                    for j in range(i, i + L):
                        used[j] = True
        return sorted(hits, key=lambda h: h[0])

    def first(self, tokens):
        hits = self.find(tokens)
        return hits[0][2] if hits else None


class Gazetteer:
    def __init__(self, path=RESOURCES / "gazetteer.json"):
        with open(path, encoding="utf-8") as f:
            self.places = json.load(f)
        self.index = {}
        for canon, info in self.places.items():
            for lang in ("en", "hi", "ta"):
                for form in info.get(lang, []):
                    self.index[" ".join(fold(t) for t in tokenize(form))] = canon
        self.max_len = max(len(k.split()) for k in self.index)

    def _lookup_seq(self, toks):
        key = " ".join(fold(t) for t in toks)
        if key in self.index:
            return self.index[key]
        if toks and token_script(toks[-1]) == "ta":
            for cand in tamil_stem_candidates(toks[-1])[1:]:
                key = " ".join([fold(t) for t in toks[:-1]] + [fold(cand)])
                if key in self.index:
                    return self.index[key]
        return None

    def lookup(self, text_or_tokens):
        """Canonical place for a slot value, trying sub-spans so 'சென்னை நகரில்' or 'मुंबई महाराष्ट्र' resolve."""
        toks = tokenize(text_or_tokens) if isinstance(text_or_tokens, str) else list(text_or_tokens)
        for L in range(min(self.max_len, len(toks)), 0, -1):
            for i in range(len(toks) - L + 1):
                canon = self._lookup_seq(toks[i : i + L])
                if canon:
                    return canon
        return None

    def find_spans(self, tokens):
        """All (start, end, canonical) place mentions in an utterance."""
        spans, i = [], 0
        while i < len(tokens):
            for L in range(min(self.max_len, len(tokens) - i), 0, -1):
                canon = self._lookup_seq(tokens[i : i + L])
                if canon:
                    spans.append((i, i + L, canon))
                    i += L
                    break
            else:
                i += 1
        return spans

    def coords(self, canon):
        info = self.places.get(canon)
        return (info["lat"], info["lon"]) if info else None

    def display(self, canon, lang):
        info = self.places.get(canon)
        if not info:
            return canon
        if lang == "en":
            return canon
        return info.get(lang, [canon])[0]

    def surface_forms(self, canon, lang):
        return self.places[canon].get(lang, [])


class CanonicalLexicon:
    def __init__(self, path=RESOURCES / "canonical.json"):
        with open(path, encoding="utf-8") as f:
            self.raw = json.load(f)
        self.date = self._matcher("date")
        self.timeofday = self._matcher("timeofday")
        self.descriptor = self._matcher("descriptor")
        self.home = FormMatcher(
            [(tokenize(f), "home", lang) for lang, forms in self.raw["home"].items() for f in forms]
        )
        self.followup = FormMatcher(
            [(tokenize(f), "followup", lang) for lang, forms in self.raw["followup_markers"].items() for f in forms]
        )
        self.past_markers = {fold(m) for forms in self.raw["past_markers"].values() for m in forms}

    def _matcher(self, key):
        entries = []
        for e in self.raw[key]:
            for lang in ("en", "hi", "ta"):
                for form in e.get(lang, []):
                    toks = tokenize(form)
                    if toks:
                        entries.append((toks, e["value"], lang))
        return FormMatcher(entries)


@lru_cache(maxsize=1)
def gazetteer() -> Gazetteer:
    return Gazetteer()


@lru_cache(maxsize=1)
def canonical_lexicon() -> CanonicalLexicon:
    return CanonicalLexicon()
