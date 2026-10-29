"""Optional speech input via Whisper (transformers pipeline).

WAV bytes (the web UI) need only numpy. File paths other than .wav still need
`pip install soundfile`.
"""

from functools import lru_cache
from pathlib import Path

WHISPER_LANGS = {"en": "english", "hi": "hindi", "ta": "tamil"}
DEFAULT_MODEL = "openai/whisper-small"


@lru_cache(maxsize=2)
def _pipeline(model_name: str):
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model=model_name)


def warmup(model_name: str = DEFAULT_MODEL):
    _pipeline(model_name)
    return model_name


def _detect_lang(pipe, audio, sr) -> str:
    """Best of English / Hindi / Tamil from Whisper's language head.

    Other language tokens are ignored, so a nearby language (for example Urdu)
    cannot knock the result out of the three languages this assistant speaks.
    """
    import torch

    model = pipe.model
    device = next(model.parameters()).device
    feats = pipe.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        lang_ids = model.detect_language(feats[:, :, :3000])
    token_id = int(lang_ids.view(-1)[0].item())
    raw = pipe.tokenizer.decode([token_id])
    code = raw.replace("<|", "").replace("|>", "").strip().lower()
    if code in WHISPER_LANGS:
        return code
    # detect_language's global argmax left the set. Score only en / hi / ta.
    gen = model.generation_config
    wanted = []
    for short, name in WHISPER_LANGS.items():
        token = f"<|{short}|>"
        idx = gen.lang_to_id.get(token)
        if idx is None:
            idx = gen.lang_to_id.get(f"<|{name}|>")
        if idx is not None:
            wanted.append((short, idx))
    if not wanted:
        return ""
    start = torch.ones((feats.shape[0], 1), device=device, dtype=torch.long) * gen.decoder_start_token_id
    with torch.no_grad():
        logits = model(input_features=feats[:, :, :3000], decoder_input_ids=start, use_cache=False).logits[:, -1, :]
    best = max(wanted, key=lambda item: float(logits[0, item[1]]))
    return best[0]


def _generate_kwargs(lang: str) -> dict:
    # One greedy pass. Beam search on Whisper-small is what made the mic feel stuck,
    # and a text prompt loops. task=transcribe stops Hindi being translated into English.
    kwargs = {
        "task": "transcribe",
        "num_beams": 1,
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 64,
        "condition_on_prev_tokens": False,
    }
    if lang in WHISPER_LANGS:
        kwargs["language"] = WHISPER_LANGS[lang]
    return kwargs


def _decode(pipe, audio, sr, lang: str) -> str:
    kwargs = _generate_kwargs(lang)
    try:
        out = pipe({"raw": audio, "sampling_rate": sr}, generate_kwargs=kwargs)
    except Exception:
        tiny = {"task": "transcribe", "num_beams": 1, "max_new_tokens": 64}
        if lang in WHISPER_LANGS:
            tiny["language"] = WHISPER_LANGS[lang]
        out = pipe({"raw": audio, "sampling_rate": sr}, generate_kwargs=tiny)
    return (out.get("text") or "").strip()


def _has_native_script(text: str, lang: str) -> bool:
    if lang == "hi":
        return any("\u0900" <= ch <= "\u097f" for ch in text)
    if lang == "ta":
        return any("\u0b80" <= ch <= "\u0bff" for ch in text)
    return False


def transcribe_wav(audio, sr, lang: str = None, model_name: str = DEFAULT_MODEL):
    """Return (text, language code). `lang` forces en/hi/ta.

    Auto mode decodes once and lets Whisper pick the language inside that pass.
    A second pass runs only when the text is still Latin but looks like Hindi or Tamil,
    which is the "aaj mausam kaisa hai" → English failure.
    """
    pipe = _pipeline(model_name)
    forced = lang in WHISPER_LANGS
    text = _decode(pipe, audio, sr, lang if forced else None)
    if forced:
        return text, lang
    from weather_nlp.preprocessing.langid import detect_language

    guessed, _shares = detect_language(text)
    if guessed in ("hi", "ta") and not _has_native_script(text, guessed):
        text = _decode(pipe, audio, sr, guessed)
        return text, guessed
    if _has_native_script(text, "hi"):
        return text, "hi"
    if _has_native_script(text, "ta"):
        return text, "ta"
    return text, guessed if guessed in WHISPER_LANGS else "en"


def transcribe_wav_bytes(data: bytes, lang: str = None, model_name: str = DEFAULT_MODEL) -> str:
    import io
    import wave

    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    text, _used = transcribe_wav(audio, sr, lang=lang, model_name=model_name)
    return text


def transcribe_wav_bytes_meta(data: bytes, lang: str = None, model_name: str = DEFAULT_MODEL):
    import io
    import wave

    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return transcribe_wav(audio, sr, lang=lang, model_name=model_name)


def transcribe(path: str, lang: str = None, model_name: str = DEFAULT_MODEL) -> str:
    p = Path(path)
    if p.suffix.lower() == ".wav":
        return transcribe_wav_bytes(p.read_bytes(), lang=lang, model_name=model_name)
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    text, _used = transcribe_wav(audio, sr, lang=lang, model_name=model_name)
    return text
