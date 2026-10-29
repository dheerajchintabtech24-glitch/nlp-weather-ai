"""Joint intent classification + slot filling (+ optional dialect ID) on a multilingual encoder."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class JointNLUModel(nn.Module):
    def __init__(
        self,
        encoder_name_or_path: str,
        num_intents: int,
        num_slots: int,
        num_dialects: int = 0,
        dropout: float = 0.1,
        layer_mix: bool = False,
    ):
        """``layer_mix``: features are a learned softmax-weighted mix of all encoder layers followed by
        LayerNorm (ELMo-style scalar mix). Needed for MuRIL, whose released checkpoint has a
        degenerate final layer (LayerNorm gain ~0.015, token vectors with cosine ~0.98), which
        starves a head that reads only ``last_hidden_state``."""
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        hidden = self.encoder.config.hidden_size
        self.layer_mix = layer_mix
        if layer_mix:
            self.mix_weights = nn.Parameter(torch.zeros(self.encoder.config.num_hidden_layers + 1))
            self.mix_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, num_intents)
        self.slot_head = nn.Linear(hidden, num_slots)
        self.dialect_head = nn.Linear(hidden, num_dialects) if num_dialects else None

    def features(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=self.layer_mix)
        if not self.layer_mix:
            return out.last_hidden_state
        layers = torch.stack(out.hidden_states, 0)
        w = self.mix_weights.softmax(0).view(-1, 1, 1, 1)
        return self.mix_norm((w * layers).sum(0))

    def forward(self, input_ids, attention_mask):
        hidden = self.dropout(self.features(input_ids, attention_mask))
        cls = hidden[:, 0]
        out = {"intent": self.intent_head(cls), "slots": self.slot_head(hidden)}
        if self.dialect_head is not None:
            out["dialect"] = self.dialect_head(cls)
        return out


def encode(tokenizer, batch_tokens, max_length: int = 64):
    """Tokenise pre-split words; returns encodings and, per example, the first-subword index of each word."""
    enc = tokenizer(
        batch_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    first_sub = []
    for i in range(len(batch_tokens)):
        seen, idx = set(), {}
        for pos, w in enumerate(enc.word_ids(i)):
            if w is not None and w not in seen:
                seen.add(w)
                idx[w] = pos
        first_sub.append(idx)
    return enc, first_sub


def slot_targets(first_sub, batch_tags, slot2id, seq_len):
    y = torch.full((len(batch_tags), seq_len), -100, dtype=torch.long)
    for i, tags in enumerate(batch_tags):
        for w, pos in first_sub[i].items():
            y[i, pos] = slot2id.get(tags[w], slot2id["O"])
    return y


def save_model(model: JointNLUModel, tokenizer, out_dir: Path, meta: dict):
    out_dir = Path(out_dir)
    (out_dir / "encoder").mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out_dir / "encoder")
    tokenizer.save_pretrained(out_dir / "encoder")
    heads = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    torch.save(heads, out_dir / "heads.pt")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_model(model_dir: Path, device: str = "cpu"):
    model_dir = Path(model_dir)
    with open(model_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    model = JointNLUModel(
        str(model_dir / "encoder"),
        len(meta["intents"]),
        len(meta["slots"]),
        len(meta.get("dialects") or []),
        layer_mix=meta.get("layer_mix", False),
    )
    model.load_state_dict(torch.load(model_dir / "heads.pt", map_location=device), strict=False)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "encoder")
    return model, tokenizer, meta
