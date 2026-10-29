"""Fine-tuning loop for the joint intent/slot(/dialect) model."""

import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

from ..config import dialect_label
from .data import build_label_maps
from .metrics import nlu_report, repair_bio
from .model import JointNLUModel, encode, save_model, slot_targets


def _batches(rows, size, shuffle, rng):
    idx = list(range(len(rows)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        yield [rows[j] for j in idx[i : i + size]]


@torch.no_grad()
def predict_rows(model, tokenizer, rows, intents, slots, dialects=None, batch_size=64, max_length=64, device="cpu"):
    model.eval()
    out = []
    for batch in _batches(rows, batch_size, False, None):
        toks = [r["tokens"] for r in batch]
        enc, first_sub = encode(tokenizer, toks, max_length)
        res = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        iprob = res["intent"].softmax(-1)
        sl = res["slots"].argmax(-1).cpu().tolist()
        for i, r in enumerate(batch):
            tags = [slots[sl[i][first_sub[i][w]]] if w in first_sub[i] else "O" for w in range(len(r["tokens"]))]
            conf, k = iprob[i].max(-1)
            pred = {"intent": intents[int(k)], "intent_conf": float(conf), "tags": repair_bio(tags)}
            if dialects and "dialect" in res:
                dp = res["dialect"][i].softmax(-1)
                c, d = dp.max(-1)
                pred["dialect"], pred["dialect_conf"] = dialects[int(d)], float(c)
            out.append(pred)
    return out


def evaluate_rows(model, tokenizer, rows, intents, slots, dialects=None, **kw):
    preds = predict_rows(model, tokenizer, rows, intents, slots, dialects, **kw)
    groups = defaultdict(list)
    for r, p in zip(rows, preds):
        groups["all"].append((r, p))
        groups[dialect_label(r["lang"], r["dialect"])].append((r, p))
    report = {}
    for g, pairs in groups.items():
        rs, ps = zip(*pairs)
        report[g] = nlu_report(
            [r["intent"] for r in rs], [p["intent"] for p in ps], [r["tags"] for r in rs], [p["tags"] for p in ps]
        )
        if dialects:
            report[g]["dialect_acc"] = sum(
                p.get("dialect") == dialect_label(r["lang"], r["dialect"]) for r, p in pairs
            ) / len(pairs)
    return report, preds


def train(
    train_rows,
    dev_rows,
    encoder: str,
    out_dir: Path,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-5,
    head_lr: float = 1e-3,
    warmup: float = 0.1,
    max_length: int = 64,
    dialect_head: bool = False,
    dialect_weight: float = 0.5,
    freeze_embeddings: bool = False,
    layer_mix: bool = True,
    seed: int = 13,
    threads: int = None,
    log_every: int = 50,
):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    torch.set_num_threads(threads or os.cpu_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    intents, slots, dialects = build_label_maps(train_rows + dev_rows, dialect_head)
    intent2id = {x: i for i, x in enumerate(intents)}
    slot2id = {x: i for i, x in enumerate(slots)}
    dialect2id = {x: i for i, x in enumerate(dialects or [])}

    tokenizer = AutoTokenizer.from_pretrained(encoder)
    model = JointNLUModel(encoder, len(intents), len(slots), len(dialects or []), layer_mix=layer_mix).to(device)
    if freeze_embeddings:
        # The word-embedding matrix dominates the parameter count of large-vocabulary encoders
        # (MuRIL: 151M of 237M); freezing it makes CPU optimiser steps far cheaper.
        model.encoder.get_input_embeddings().weight.requires_grad_(False)

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = AdamW([{"params": enc_params, "lr": lr}, {"params": head_params, "lr": head_lr}], weight_decay=0.01)
    total = epochs * math.ceil(len(train_rows) / batch_size)
    warm = int(warmup * total)
    sched = LambdaLR(opt, lambda s: (s + 1) / max(warm, 1) if s < warm else max(0.0, (total - s) / max(total - warm, 1)))
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    meta = {
        "encoder": encoder,
        "intents": intents,
        "slots": slots,
        "dialects": dialects,
        "max_length": max_length,
        "layer_mix": layer_mix,
        "train_size": len(train_rows),
        "hyperparams": dict(
            epochs=epochs, batch_size=batch_size, lr=lr, head_lr=head_lr, dialect_weight=dialect_weight, freeze_embeddings=freeze_embeddings
        ),
    }
    log = {"meta": meta, "epochs": []}
    best, step, t0 = -1.0, 0, time.time()
    print(f"[train] {len(train_rows)} train / {len(dev_rows)} dev, {len(intents)} intents, {len(slots)} slot tags, {total} steps on {device}")

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in _batches(train_rows, batch_size, True, rng):
            enc, first_sub = encode(tokenizer, [r["tokens"] for r in batch], max_length)
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            y_int = torch.tensor([intent2id[r["intent"]] for r in batch], device=device)
            y_slot = slot_targets(first_sub, [r["tags"] for r in batch], slot2id, ids.shape[1]).to(device)
            out = model(ids, mask)
            loss = ce(out["intent"], y_int) + ce(out["slots"].reshape(-1, len(slots)), y_slot.reshape(-1))
            if dialects:
                y_d = torch.tensor([dialect2id[dialect_label(r["lang"], r["dialect"])] for r in batch], device=device)
                loss = loss + dialect_weight * ce(out["dialect"], y_d)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += loss.item()
            if step % log_every == 0:
                el = time.time() - t0
                print(f"[train] ep {ep} step {step}/{total} loss {running / log_every:.4f} ({el / step:.2f}s/step, eta {el / step * (total - step) / 60:.1f} min)")
                running = 0.0

        report, _ = evaluate_rows(model, tokenizer, dev_rows, intents, slots, dialects, max_length=max_length, device=device)
        score = (report["all"]["intent_acc"] + report["all"]["slot_f1"]) / 2
        log["epochs"].append({"epoch": ep, "dev": report})
        print(f"[dev] ep {ep}: intent {report['all']['intent_acc']:.4f} slotF1 {report['all']['slot_f1']:.4f} EM {report['all']['exact_match']:.4f}")
        if score > best:
            best = score
            meta["best_epoch"], meta["best_dev"] = ep, report["all"]
            save_model(model, tokenizer, out_dir, meta)
            print(f"[save] best model -> {out_dir}")

    with open(Path(out_dir) / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    return log
