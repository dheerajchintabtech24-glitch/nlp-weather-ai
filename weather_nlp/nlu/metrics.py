"""Intent accuracy, CoNLL-style span F1, value-based slot F1 and exact match."""

from collections import Counter

from ..preprocessing import fold


def bio_spans(tags):
    """Set of (slot_type, start, end); a stray I- starts a new span (lenient decoding)."""
    spans, start, cur = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != cur):
            if cur is not None:
                spans.append((cur, start, i))
            cur, start = (t[2:], i) if t != "O" else (None, None)
    return set(spans)


def repair_bio(tags):
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]
        out.append(t)
        prev = t
    return out


def slot_values(tokens, tags):
    """Counter of (slot_type, folded value) -- comparable even when tokenisations differ."""
    return Counter((s, " ".join(fold(t) for t in tokens[a:b])) for s, a, b in bio_spans(tags))


def _prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def span_f1(gold_tags, pred_tags):
    tp = n_pred = n_gold = 0
    for g, p in zip(gold_tags, pred_tags):
        gs, ps = bio_spans(g), bio_spans(p)
        tp += len(gs & ps)
        n_pred += len(ps)
        n_gold += len(gs)
    return _prf(tp, n_pred, n_gold)


def value_f1(gold_values, pred_values):
    tp = n_pred = n_gold = 0
    for g, p in zip(gold_values, pred_values):
        tp += sum((g & p).values())
        n_pred += sum(p.values())
        n_gold += sum(g.values())
    return _prf(tp, n_pred, n_gold)


def nlu_report(gold_intents, pred_intents, gold_tags=None, pred_tags=None, gold_values=None, pred_values=None):
    n = len(gold_intents)
    rep = {"n": n, "intent_acc": sum(g == p for g, p in zip(gold_intents, pred_intents)) / max(n, 1)}
    if gold_tags is not None:
        p, r, f = span_f1(gold_tags, pred_tags)
        rep.update(slot_p=p, slot_r=r, slot_f1=f)
        rep["exact_match"] = (
            sum(gi == pi and bio_spans(gt) == bio_spans(pt) for gi, pi, gt, pt in zip(gold_intents, pred_intents, gold_tags, pred_tags))
            / max(n, 1)
        )
    if gold_values is not None:
        p, r, f = value_f1(gold_values, pred_values)
        rep.update(value_p=p, value_r=r, value_f1=f)
        rep["value_exact_match"] = (
            sum(gi == pi and gv == pv for gi, pi, gv, pv in zip(gold_intents, pred_intents, gold_values, pred_values))
            / max(n, 1)
        )
    return rep
