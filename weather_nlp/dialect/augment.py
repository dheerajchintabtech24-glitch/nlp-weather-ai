"""Slot-preserving dialect augmentation.

Rewrites a standard MASSIVE utterance into a regional variety while keeping BIO tags aligned:
a replacement is only allowed inside one segment (all-O, or a single slot entity), and the
replacement tokens inherit that segment's tag (B- on the first token if the original began with B-).
"""

import random

from ..preprocessing import fold
from .lexicon import DialectLayer, DialectLexicon

PULLI = "\u0bcd"


def _segment_ok(tags):
    """True if ``tags`` is all O or one contiguous entity of one slot type."""
    if all(t == "O" for t in tags):
        return True
    if tags[0] == "O":
        return False
    slot = tags[0][2:]
    return all(t == f"I-{slot}" for t in tags[1:])


def _retag(span_tags, n):
    if span_tags[0] == "O":
        return ["O"] * n
    slot = span_tags[0][2:]
    return [span_tags[0]] + [f"I-{slot}"] * (n - 1)


class DialectAugmenter:
    def __init__(self, lexicon: DialectLexicon = None, p_replace: float = 0.85, p_particle: float = 0.4):
        self.lexicon = lexicon or DialectLexicon()
        self.p_replace = p_replace
        self.p_particle = p_particle

    def transform(self, tokens, tags, lang: str, dialect: str, rng: random.Random):
        layer: DialectLayer = self.lexicon.layer(lang, dialect)
        out_toks, out_tags, changes = [], [], 0
        i, n = 0, len(tokens)
        folded = [fold(t) for t in tokens]
        while i < n:
            done = False
            for L in range(min(layer.max_phrase_len, n - i), 1, -1):
                key = tuple(folded[i : i + L])
                if key in layer.phrases and _segment_ok(tags[i : i + L]) and rng.random() < self.p_replace:
                    var = rng.choice(layer.phrases[key])
                    out_toks += var
                    out_tags += _retag(tags[i : i + L], len(var))
                    i, changes, done = i + L, changes + 1, True
                    break
            if done:
                continue
            tok, tag = tokens[i], tags[i]
            variants = layer.words.get(folded[i])
            if variants and rng.random() < self.p_replace:
                var = rng.choice(variants)
                out_toks += var
                out_tags += _retag([tag], len(var))
                changes += var != [tok]
            else:
                new = tok
                for rule in layer.suffix_rules:
                    if rule.applies_to(tag) and rule.pattern.search(tok):
                        if rng.random() < self.p_replace:
                            new = rule.pattern.sub(rule.repl, tok)
                        break
                out_toks.append(new)
                out_tags.append(tag)
                changes += new != tok
            i += 1
        changes += self._add_particle(layer, out_toks, out_tags, rng)
        return out_toks, out_tags, changes

    def _add_particle(self, layer, toks, tags, rng):
        if not toks or rng.random() >= self.p_particle:
            return 0
        options = []
        if layer.prefix_particles:
            options.append("prefix")
        if layer.suffix_particles:
            options.append("suffix")
        if (
            layer.attach_particles
            and tags[-1] == "O"
            and not toks[-1].endswith(PULLI)
            and not any(toks[-1].endswith(a) for a in layer.attach_particles)
        ):
            options.append("attach")
        if not options:
            return 0
        mode = rng.choice(options)
        if mode == "prefix":
            p = rng.choice(layer.prefix_particles)
            toks[:0] = p
            tags[:0] = ["O"] * len(p)
        elif mode == "suffix":
            p = rng.choice(layer.suffix_particles)
            toks += p
            tags += ["O"] * len(p)
        else:
            toks[-1] = toks[-1] + rng.choice(layer.attach_particles)
        return 1

    def variants(self, record: dict, dialect: str, k: int = 1, rng: random.Random = None, max_tries: int = 6):
        """Up to ``k`` distinct dialect rewrites of a processed MASSIVE record."""
        rng = rng or random.Random(0)
        seen, out = {" ".join(record["tokens"])}, []
        for _ in range(k * max_tries):
            toks, tags, changes = self.transform(record["tokens"], record["tags"], record["lang"], dialect, rng)
            key = " ".join(toks)
            if changes and key not in seen:
                seen.add(key)
                out.append((toks, tags))
                if len(out) == k:
                    break
        return out
