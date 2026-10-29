"""Neural speech for the spoken reply. Voices follow the question language, not the dialect.

edge-tts voices: English (India), Hindi, Tamil. Dialect wording is already in the text;
these voices do not speak Kongu, Bhojpuri, Haryanvi or the other regional varieties.
"""

import asyncio
import hashlib
from pathlib import Path

from .config import CACHE_DIR

VOICES = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
}


def synthesize(text: str, lang: str, cache_dir: Path = None) -> Path:
    """Return a cached mp3 for this (language, text) pair. Requires `pip install edge-tts` and internet."""
    lang = lang if lang in VOICES else "en"
    text = " ".join((text or "").split())
    if not text:
        raise ValueError("empty speech text")
    voice = VOICES[lang]
    folder = Path(cache_dir) if cache_dir else CACHE_DIR / "tts"
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{voice}\n{text}".encode("utf-8")).hexdigest()[:20]
    path = folder / f"{lang}-{digest}.mp3"
    if path.exists() and path.stat().st_size > 500:
        return path
    try:
        import edge_tts
    except ImportError as e:
        raise RuntimeError("speech synthesis needs edge-tts (`pip install edge-tts`)") from e

    tmp = folder / f"{lang}-{digest}.tmp.mp3"

    async def _save():
        await edge_tts.Communicate(text, voice).save(str(tmp))

    # ThreadingHTTPServer workers may already have a loop; always use a fresh one.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_save())
    finally:
        loop.close()

    if not tmp.exists() or tmp.stat().st_size < 500:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("speech synthesis returned no audio")
    tmp.replace(path)
    return path
