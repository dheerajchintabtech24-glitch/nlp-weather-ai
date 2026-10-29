# Context-Aware, Dialect-Robust Weather Assistant for English, Hindi and Tamil

Implementation of the project described in *Context-Aware NLP-Based Weather AI Assistant for
Low-Resource Indian Languages* extended with **regional dialect support**
for Hindi (Bhojpuri-influenced, Haryanvi, Bambaiya) and Tamil (Chennai, Kongu, Madurai, Nellai).

```
speech (Whisper, optional)
  -> Unicode normalisation + script language ID                      preprocessing/
  -> dialect identification (char n-gram SVM)                        dialect/identifier.py
  -> joint intent + slot model (MuRIL fine-tuned on MASSIVE + dialect data)   nlu/
  -> canonicalisation: Tamil stemming + gazetteer, dates, descriptors  dialogue/canonicalize.py
  -> language-independent conversation state with carry-over          dialogue/state.py
  -> tool router -> Open-Meteo live weather                            tools/
  -> grounded response in the requested language (en / hi / ta)       nlg/responder.py
```

Dialect→standard rewriting (`dialect/normalizer.py`) is **off by default**. With `muril_dialect`
it does not improve weather frame accuracy; use `--normalize` only as an ablation (or with
`muril_std`). The char seq2seq normaliser is an optional experiment, not part of the main path.

The input language, the dialect and the reply language are independent: you can ask in Kongu Tamil,
follow up in Hindi and get the answer in English, and the state (intent, place, date, descriptor)
survives every switch because it only contains language-independent symbols.

## Where each part of the report is implemented

| Report (Sec. II-F) | Code |
|---|---|
| Pretrained multilingual speech recognition [2], [3] | `weather_nlp/asr.py` (Whisper via transformers; `/audio` in chat) |
| Language-aware preprocessing, Indic morphology instead of English stemmers [13] | `preprocessing/text.py` (NFC, nukta/chandrabindu folding), `preprocessing/langid.py`, `preprocessing/stemmers.py` (Tamil analysis-by-lookup stemmer, Hindi light stemmer) |
| Multilingual encoder [1], [6] fine-tuned on MASSIVE hi/ta/en [5] for joint intent + slots | `nlu/model.py`, `nlu/train.py`, `scripts/train_nlu.py` |
| Conversation-state manager with shared intent symbol and canonical entities | `dialogue/canonicalize.py`, `dialogue/state.py` |
| Tool router depending only on language-independent symbols, grounded answer [11] | `tools/router.py`, `tools/weather.py` |
| Rendering in the requested language [12] | `nlg/responder.py` (grounded templates; every value comes from the API payload) |
| Evaluation: cross-lingual follow-ups vs. same-language control | `evaluation/followup.py`, `scripts/evaluate_context.py` |

**Dialect extension (new):**

| Component | Code |
|---|---|
| Dialect lexicons (phrase / word maps, Tamil suffix rules, discourse particles) | `resources/hi_dialects.json`, `resources/ta_dialects.json` |
| Slot-preserving dialect augmentation of MASSIVE | `dialect/augment.py`, `scripts/prepare_data.py` |
| Dialect identification: char n-gram TF-IDF + SVM / LogReg / NB | `dialect/identifier.py`, `scripts/train_dialect_id.py` |
| Optional lexicon normaliser (ablation / for `muril_std`; gated by dialect ID) | `dialect/normalizer.py`, `chat.py --normalize` |
| Optional char BiGRU+attention seq2seq normaliser (experiment, not used in chat) | `dialect/seq2seq.py`, `scripts/train_normalizer.py` |
| Optional multi-task dialect head on the NLU encoder | `train_nlu.py --dialect-head` |
| Gazetteer with historic/colloquial names (Madras, Kovai, Nellai, Banaras, ...) | `resources/gazetteer.json` |
| Dialect forms in the canonical lexicon (काल्ह, बरखा, मींह, இன்னிக்கு, ஜில்லுனு, ...) | `resources/canonical.json` |

## Setup

```powershell
pip install -r requirements.txt
python scripts/download_data.py      # MASSIVE 1.1 (CC BY 4.0) -> data/raw
python scripts/prepare_data.py       # BIO conversion + dialect data -> data/processed
python -m pytest -q tests
```

## Training

```powershell
python scripts/train_dialect_id.py                                     # ~2 min, CPU
python scripts/train_nlu.py --name muril_dialect                       # main model (use this in chat)
python scripts/train_nlu.py --name muril_std --no-dialects             # ablation baseline
# optional experiments:
python scripts/train_nlu.py --name xlmr_dialect --encoder xlm-roberta-base
python scripts/train_nlu.py --name muril_multitask --dialect-head
python scripts/train_normalizer.py --epochs 10                         # seq2seq normaliser (not needed for chat)
```

On a laptop CPU a MuRIL step takes ~3.5 s (with `--freeze-embeddings`), so the full NLU runs
(~60k sentences x 3 epochs) take hours; use `notebooks/train_on_colab.ipynb` (T4 GPU, ~20 min per
model). The local models in this repo were trained with
`--max-per-file 800 --epochs 3 --freeze-embeddings` (keeps every weather sentence, subsamples the
other intents); the standard-only baseline gets the same number of sentences and updates
(`--no-dialects --max-per-file 2667`).

**MuRIL final-layer degeneracy.** The released `google/muril-base-cased` checkpoint has a final
LayerNorm gain of ~0.015, so its `last_hidden_state` vectors are tiny (mean |h| ~0.005) and nearly
identical across tokens (mean pairwise cosine 0.98 at layer 12 vs 0.48 at layer 6). A head on the
last layer barely learns slot tagging (0% slot-token accuracy after 40 steps when overfitting 32
sentences). The NLU therefore reads a learned softmax-weighted mix of all layers followed by
LayerNorm (`layer_mix`, on by default), which reaches 100% on the same test in 16 steps.

## Evaluation

```powershell
python scripts/evaluate_nlu.py --models rules muril_std muril_dialect               # main weather numbers
python scripts/evaluate_nlu.py --models rules muril_std muril_dialect --normalize   # +norm ablation columns
python scripts/evaluate_nlu.py --models muril_dialect --scope all                   # all 60 MASSIVE intents
python scripts/evaluate_context.py --models rules muril_dialect                     # follow-up benchmark
```

Results are written to `reports/` as Markdown and JSON.

## Presentation UI

```powershell
python scripts/serve.py                      # http://127.0.0.1:7860  (loads muril_dialect)
python scripts/serve.py --nlu rules --offline
```

The browser UI (**NLP Weather AI**) is a simple chat: type or use the mic in English, Hindi or
Tamil, see language / dialect / slots / weather on the side, and hear the reply. The written reply
can still use another language. **Speak in the question's language** reads a spoken form of the
answer (numbers as words, then dialect wording from the existing lexicons) with an English, Hindi
or Tamil neural voice. The voice follows the question language, not the dialect: Kongu or Bhojpuri
changes the words, not the speaker.

The first Whisper load downloads `openai/whisper-small` (~500 MB) and can take a few minutes.
Until the Whisper pill lights up, the mic is disabled. Use `--asr openai/whisper-tiny` for a
faster CPU demo.

## Chat

Models already in `models/` (`muril_dialect`, dialect ID). No normaliser needed.

```powershell
python scripts/chat.py --nlu muril_dialect            # recommended: live Open-Meteo
python scripts/chat.py --nlu muril_dialect --offline --demo   # offline scripted demo (no internet)
python scripts/chat.py --nlu muril_std --normalize    # ablation: standard NLU + lexicon rewrite
python scripts/chat.py --nlu rules --offline --demo   # rule baseline, no transformer
```

Inside the chat: `/lang ta` (reply language), `/audio file.wav`, `/state`, `/debug`, `/reset`.

## Adding real dialect data (important)

The dialect training/test data is **synthetic**: MASSIVE sentences rewritten with the seed lexicons.
The lexicons were written from documented dialect features and must be checked by native speakers.
Synthetic test sets also over-estimate performance, because train and test share the same rewrite
rules. To report credible dialect numbers:

1. Have speakers from each region rewrite standard weather sentences in their dialect, keeping the
   slot brackets, in `data/dialect/collected/*.csv` (columns as in `TEMPLATE.csv`:
   `lang,dialect,annot_utt,intent,standard_annot_utt,region,annotator`).
2. Have a second annotator label a subset and report agreement (Cohen's kappa).
3. Re-run `prepare_data.py`; the collected rows become a separate `(real)` test group in every
   evaluation table and are never used for training.

## Results

Local CPU training (i7-1355U, 16 GB RAM, no GPU). Dialect ID and the two NLU models are
trained. The deployed path is `muril_dialect` **without** dialect→standard normalisation.
Full tables live in `reports/`. Dialect numbers are **synthetic** (see above) and over-estimate
real dialect performance.

### Dialect identification (test, n = 5686)

Char n-gram TF-IDF (`char_wb` 1–5 + word 1–2) restricted to the script-detected language.

| Classifier | Macro-F1 | Accuracy |
|---|---|---|
| SVM (linear) | 0.923 | 0.922 |
| Logistic regression | 0.923 | 0.922 |
| Naive Bayes | 0.850 | 0.850 |

Hindi dialects score 0.96–0.99 F1. The main error is Chennai ↔ Madurai (Chennai F1 0.80, Madurai
F1 0.79) because the Madurai lexicon is thin. Saved to `models/dialect_id/`.

### Weather NLU (MASSIVE weather + dialect test, n = 1543)

Both MuRIL models: 3 epochs, ~8 000 train sentences, `--freeze-embeddings`, `layer_mix`.
`muril_dialect` is trained on standard + 7 dialect varieties; `muril_std` is the matched-budget
standard-only ablation. Surface slot F1 is omitted after normalisation because the gold tags still
use dialect forms; `canon_f1` / `frame_acc` are the task-level scores.

| System | Intent | Span F1 | Canon F1 | Frame acc |
|---|---|---|---|---|
| rules | 86.7 | 67.8 | 87.3 | 64.3 |
| muril_std | 93.3 | 76.3 | 92.7 | 79.8 |
| muril_std + lexicon-norm | 95.1 | — | 94.5 | 84.1 |
| **muril_dialect** | **97.2** | **81.8** | **95.8** | **88.7** |
| muril_dialect + lexicon-norm | 96.9 | — | 95.8 | 88.7 |

Dialect training closes the Haryanvi gap that the standard-only model leaves (intent 84.6 → 98.1)
and lifts every Tamil dialect by ~6 span-F1 points. Lexicon normalisation helps `muril_std` on
Hindi (frame acc 74.1 → 86.6) and **does not help `muril_dialect`**, which already saw the dialect
forms — so chat defaults to `normalize=off`. Oracle dialect vs predicted dialect differs by < 0.5 points.

### All 60 MASSIVE intents (n = 12 859)

`muril_std` beats `muril_dialect` here (intent 79.7 vs 71.9) because weather sentences are always
kept when subsampling, so the dialect run spends a larger share of its 8 000 examples on weather
and sees fewer of the other 59 intents. Use `muril_dialect` for the weather assistant; train
without `--max-per-file` (or on GPU via `notebooks/train_on_colab.ipynb`) if you need the full
MASSIVE label set.

### Cross-lingual / cross-dialect follow-ups (n = 4 500)

Two-turn conversations: MASSIVE weather turn 1, then a follow-up in every (lang, dialect) pair.
Joint goal accuracy (JGA) requires the full canonical frame (intent + place + date + …) after
carry-over. Without the conversation-state manager, JGA collapses to 0–9 %.

| System | JGA | Intent | Target acc | Carry acc |
|---|---|---|---|---|
| rules | 70.7 | 86.3 | 99.6 | 84.8 |
| muril_std | 78.7 | 94.2 | 97.3 | 94.0 |
| **muril_dialect** | **83.2** | **97.0** | 97.3 | **96.7** |
| muril_dialect / oracle turn 1 | 96.6 | 99.4 | 97.3 | 99.5 |
| muril_dialect / no context | 9.2 | 28.7 | 97.3 | 31.6 |

Conditioned on the follow-up language (`muril_dialect`):

| Condition | JGA |
|---|---|
| same language, standard | 82.2 |
| cross-lingual, standard | 78.8 |
| same language, dialect | 84.3 |
| cross-lingual, dialect | 84.8 |

The remaining JGA gap vs the oracle-turn-1 upper bound (~13 points) is almost entirely turn-1
place/date errors that then poison carry-over. English follow-ups are weaker (L2=en JGA 66)
because several English templates put the date in a form the NLU has not seen as a slot.

### Seq2seq normaliser (optional experiment)

8-epoch char BiGRU on 12k synthetic pairs. The lexicon still wins, so chat does not use this model.

| System | Exact match | CER | chrF |
|---|---|---|---|
| identity | 0.0 | 15.1 | 88.8 |
| lexicon (gold dialect) | **78.9** | **1.7** | **97.0** |
| lexicon (predicted dialect) | 77.0 | 1.9 | 96.7 |
| seq2seq (gold dialect) | 73.0 | 9.4 | 93.2 |
| seq2seq (predicted dialect) | 72.2 | 9.1 | 93.3 |

## Caveats

- Dialect train/test data is lexicon-rewritten MASSIVE, not field recordings. Native-speaker
  collection (`data/dialect/collected/`) is the next step before claiming real dialect robustness.
- Follow-up utterances are generated from templates, not from users.
- Local NLU training froze the embedding layer and used 8 k sentences / 3 epochs. Full-data GPU
  training will move the all-intent numbers; the weather ranking is not expected to change.
- Tamil response templates never inflect place names (`X பகுதியில்`) because a morphological
  generator is out of scope.
- Open-Meteo geocoding is the fallback for places outside the ~50-city gazetteer; the rule NLU
  cannot tag those places at all.
