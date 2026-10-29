"""Inference wrappers: the fine-tuned transformer NLU and a lexicon/rule baseline with the same interface."""

from pathlib import Path

import torch

from ..config import WEATHER_INTENT
from ..knowledge import canonical_lexicon, gazetteer
from ..preprocessing import fold
from .metrics import bio_spans
from .model import load_model
from .train import predict_rows


def spans_from_tags(tokens, tags):
    return [
        {"type": s, "value": " ".join(tokens[a:b]), "start": a, "end": b}
        for s, a, b in sorted(bio_spans(tags), key=lambda x: x[1])
    ]


class TransformerNLU:
    name = "transformer"

    def __init__(self, model_dir: Path, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer, self.meta = load_model(model_dir, self.device)
        self.model_dir = Path(model_dir)
        self.name = self.model_dir.name

    def predict_batch(self, token_lists):
        rows = [{"tokens": t} for t in token_lists]
        preds = predict_rows(
            self.model,
            self.tokenizer,
            rows,
            self.meta["intents"],
            self.meta["slots"],
            self.meta.get("dialects"),
            max_length=self.meta.get("max_length", 64),
            device=self.device,
        )
        for toks, p in zip(token_lists, preds):
            p["slots"] = spans_from_tags(toks, p["tags"])
        return preds

    def predict(self, tokens):
        return self.predict_batch([tokens])[0]


_WEATHER_WORDS = {
    fold(w)
    for w in (
        "weather forecast climate mausam mosam mausum baarish barish barkha barsaat "
        "garmi sardi thandi temperature rain rainy cloudy "
        "मौसम पूर्वानुमान मोसम बारिश बरखा "
        "வானிலை முன்னறிவிப்பு காலநிலை வெதர் மழை"
    ).split()
}


class RuleNLU:
    """Lexicon baseline: weather keyword spotting + gazetteer/lexicon slot tagging."""

    name = "rules"

    def predict(self, tokens):
        lex, gaz = canonical_lexicon(), gazetteer()
        tags = ["O"] * len(tokens)

        def mark(a, b, slot):
            if all(t == "O" for t in tags[a:b]):
                tags[a] = f"B-{slot}"
                for j in range(a + 1, b):
                    tags[j] = f"I-{slot}"

        for a, b, _ in gaz.find_spans(tokens):
            mark(a, b, "place_name")
        for a, b, _, _ in lex.date.find(tokens):
            mark(a, b, "date")
        for a, b, _, _ in lex.timeofday.find(tokens):
            mark(a, b, "timeofday")
        descr = lex.descriptor.find(tokens)
        for a, b, _, _ in descr:
            mark(a, b, "weather_descriptor")
        is_weather = bool(descr) or any(fold(t) in _WEATHER_WORDS or fold(t).startswith("வானிலை") for t in tokens)
        return {
            "intent": WEATHER_INTENT if is_weather else "other",
            "intent_conf": 0.9 if is_weather else 0.5,
            "tags": tags,
            "slots": spans_from_tags(tokens, tags),
        }

    def predict_batch(self, token_lists):
        return [self.predict(t) for t in token_lists]
