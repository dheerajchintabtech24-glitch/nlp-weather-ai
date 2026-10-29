"""Road distance and a weather-adjusted travel time between two canonical places.

The dialogue state already stores the destination in a language-independent place slot, so
"how long will it take to get there?" works after a Hindi, Tamil or English weather turn.
Distance comes from a driving route when the network is up, otherwise from a haversine
road estimate. Rain, storms, fog, snow and strong wind stretch that duration; the numbers
in the reply are these computed values, not a language-model guess.
"""

import math

import requests

from ..knowledge import gazetteer
from ..preprocessing import fold, tokenize

# Phrases that mean a journey question rather than a weather question.
_TRIP_CUES = (
    "how long will it take",
    "how long to get",
    "how long to reach",
    "how long to go",
    "how long to",
    "how much time",
    "how far",
    "time to reach",
    "time to get",
    "time to go",
    "get there",
    "go there",
    "drive time",
    "distance",
    "कितना समय",
    "कितनी देर",
    "कितना देर",
    "कितना टाइम",
    "कितना दूर",
    "जाने में",
    "पहुंचने",
    "पहुँचने",
    "दूरी",
    "எவ்வளவு நேரம்",
    "எவ்ளோ நேரம்",
    "எத்தனை மணி",
    "எவ்வளவு தூரம்",
    "kitna time",
    "kitni der",
    "kitna dur",
    "pahunchne",
)
_FROM_BEFORE = {"from"}
_FROM_AFTER = {"से", "இருந்து", "से"}
_TO_BEFORE = {"to", "को", "வரை", "க்கு"}
_THERE = {"there", "वहाँ", "वहां", "वहा", "उहाँ", "उहां", "उधर", "उदर", "उधर", "அங்க", "அங்கே", "அங்கு"}
_SET_ORIGIN = (
    "i am in",
    "i'm in",
    "i’m in",
    "im in",
    "i live in",
    "start from",
    "starting from",
    "मेरी जगह",
    "मेरा शहर",
)

ROAD_FACTOR = 1.35
ROAD_KMH = 48.0
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def weather_slowdown(day: dict) -> tuple:
    """Return (duration multiplier, short reason code) from one forecast day."""
    from ..nlg.responder import condition_group

    group = condition_group(int(day.get("code") or 0))
    pop = day.get("pop") or 0
    wind = day.get("wind") or 0
    if group in ("thunderstorm", "snow") or pop >= 70:
        return 1.45, group if group in ("thunderstorm", "snow") else "rain"
    if group in ("rain", "showers", "drizzle") or pop >= 40:
        return 1.25, group if group in ("rain", "showers", "drizzle") else "rain"
    if group == "fog" or wind >= 40:
        return 1.2, "fog" if group == "fog" else "wind"
    return 1.0, "clear"


def driving_minutes(origin: dict, dest: dict, offline: bool = False) -> tuple:
    """Return (km, minutes, source). Minutes are the clear-weather driving time."""
    if not offline:
        try:
            url = OSRM_URL.format(lon1=origin["lon"], lat1=origin["lat"], lon2=dest["lon"], lat2=dest["lat"])
            r = requests.get(url, params={"overview": "false"}, timeout=2.5)
            r.raise_for_status()
            route = (r.json().get("routes") or [None])[0]
            if route and route.get("distance"):
                return round(route["distance"] / 1000.0), route["duration"] / 60.0, "osrm"
        except (requests.RequestException, ValueError, KeyError, IndexError):
            pass
    straight = haversine_km(origin["lat"], origin["lon"], dest["lat"], dest["lon"])
    road = straight * ROAD_FACTOR
    return round(road), road / ROAD_KMH * 60.0, "haversine"


def estimate(origin: dict, dest: dict, day: dict, offline: bool = False) -> dict:
    km, base_min, source = driving_minutes(origin, dest, offline=offline)
    factor, reason = weather_slowdown(day)
    return {
        "km": int(km),
        "minutes": int(round(base_min * factor)),
        "base_minutes": int(round(base_min)),
        "factor": factor,
        "reason": reason,
        "source": source,
    }


def _span_record(tokens, a, b, canon):
    return {"canonical": canon, "surface": " ".join(tokens[a:b])}


def split_places(tokens):
    """Return (origin, destination) place dicts named in the utterance. Either may be None."""
    spans = gazetteer().find_spans(tokens)
    origins, dests, plain = [], [], []
    for a, b, canon in spans:
        rec = _span_record(tokens, a, b, canon)
        before = [fold(t) for t in tokens[max(0, a - 3) : a]]
        after = [fold(t) for t in tokens[b : b + 2]]
        if any(t in _FROM_BEFORE for t in before) or any(t in _FROM_AFTER for t in after):
            origins.append(rec)
        elif any(t in _TO_BEFORE for t in before):
            dests.append(rec)
        else:
            plain.append(rec)
    if not origins and len(plain) >= 2:
        origins.append(plain.pop(0))
    dests = dests + plain
    return (origins[-1] if origins else None), (dests[-1] if dests else None)


def mentions_there(tokens) -> bool:
    return any(fold(t) in _THERE for t in tokens)


def is_trip(text: str) -> bool:
    hay = fold(text)
    return any(fold(cue) in hay for cue in _TRIP_CUES)


def is_set_origin(text: str, tokens) -> bool:
    hay = fold(text)
    if any(fold(cue) in hay for cue in _SET_ORIGIN):
        return True
    # "मैं दिल्ली में हूँ" / "मैं दिल्ली में हूं" names a home city, not a forecast.
    if "मैं" in tokens and "में" in tokens and any(fold(t).startswith("हू") for t in tokens):
        return True
    if any(t == "நான்" for t in tokens) and any(t.startswith("இரு") for t in tokens):
        return True
    return False


def classify(text: str):
    """Return ('trip'|'set_origin'|None, origin, destination)."""
    tokens = tokenize(text)
    origin, dest = split_places(tokens)
    if is_trip(text):
        return "trip", origin, dest
    if is_set_origin(text, tokens) and origin is None and dest is not None and len(gazetteer().find_spans(tokens)) == 1:
        return "set_origin", dest, None
    if is_set_origin(text, tokens) and origin is not None and dest is None:
        return "set_origin", origin, None
    return None, None, None
