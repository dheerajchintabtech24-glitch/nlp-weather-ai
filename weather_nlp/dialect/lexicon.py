"""Dialect lexicon: phrase/word maps, suffix rules and particles per (language, dialect)."""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from ..config import DIALECTS, RESOURCES
from ..preprocessing import fold, tokenize


@dataclass
class SuffixRule:
    pattern: re.Pattern
    repl: str
    tags: tuple = ()

    def applies_to(self, tag: str) -> bool:
        if not self.tags or "*" in self.tags:
            return True
        slot = "O" if tag == "O" else tag[2:]
        return slot in self.tags


@dataclass
class DialectLayer:
    lang: str
    dialect: str
    description: str = ""
    phrases: dict = field(default_factory=dict)  # tuple(folded std tokens) -> [variant token lists]
    words: dict = field(default_factory=dict)  # folded std token -> [variant token lists]
    suffix_rules: list = field(default_factory=list)
    prefix_particles: list = field(default_factory=list)
    suffix_particles: list = field(default_factory=list)
    attach_particles: list = field(default_factory=list)

    @property
    def max_phrase_len(self) -> int:
        return max((len(k) for k in self.phrases), default=1)


def _variants(values):
    return [tokenize(v) for v in values if tokenize(v)]


def _merge(layer: DialectLayer, block: dict, override: bool):
    for std, vals in block.get("phrases", {}).items():
        key = tuple(fold(t) for t in tokenize(std))
        if override or key not in layer.phrases:
            layer.phrases[key] = _variants(vals)
    for std, vals in block.get("words", {}).items():
        toks = tokenize(std)
        if len(toks) != 1:
            layer.phrases[tuple(fold(t) for t in toks)] = _variants(vals)
            continue
        if override or fold(toks[0]) not in layer.words:
            layer.words[fold(toks[0])] = _variants(vals)
    for r in block.get("suffix_rules", []):
        layer.suffix_rules.append(SuffixRule(re.compile(r["pattern"]), r["repl"], tuple(r.get("tags", ()))))
    parts = block.get("particles", {})
    layer.prefix_particles += _variants(parts.get("prefix", []))
    layer.suffix_particles += _variants(parts.get("suffix", []))
    layer.attach_particles += [p for p in parts.get("attach", []) if p]


class DialectLexicon:
    """Loads ``resources/{lang}_dialects.json``; a dialect layer = shared spoken layer + its own markers."""

    def __init__(self, resources=RESOURCES):
        self.raw = {}
        for lang in DIALECTS:
            path = resources / f"{lang}_dialects.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self.raw[lang] = json.load(f)

    def dialects(self, lang: str):
        return [d for d in DIALECTS.get(lang, []) if d != "standard" and d in self.raw.get(lang, {}).get("dialects", {})]

    @lru_cache(maxsize=None)
    def layer(self, lang: str, dialect: str) -> DialectLayer:
        spec = self.raw[lang]
        block = spec["dialects"][dialect]
        layer = DialectLayer(lang, dialect, block.get("description", ""))
        _merge(layer, block, override=True)
        _merge(layer, spec.get("common", {}), override=False)
        return layer

    def describe(self, lang: str, dialect: str) -> str:
        if dialect == "standard":
            return "Standard / written"
        return self.layer(lang, dialect).description
