import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows consoles default to a legacy code page that cannot print Devanagari/Tamil.
for _name in ("stdout", "stderr"):
    _s = getattr(sys, _name)
    if hasattr(_s, "buffer") and (_s.encoding or "").lower() != "utf-8":
        setattr(sys, _name, io.TextIOWrapper(_s.buffer, encoding="utf-8", errors="replace", line_buffering=True))
