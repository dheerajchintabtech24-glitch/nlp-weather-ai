"""Live weather via Open-Meteo (no API key), returning a structured payload for grounded generation."""

import datetime as dt
import hashlib
import json
import time

import requests

from ..config import CACHE_DIR, DEFAULT_LOCATION, FORECAST_DAYS
from ..dialogue.canonicalize import HOME
from ..knowledge import gazetteer
from ..preprocessing import tamil_stem_candidates, token_script, tokenize

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

PARTS_OF_DAY = {"morning": range(6, 12), "afternoon": range(12, 17), "evening": range(17, 21), "night": range(21, 24)}


class WeatherError(Exception):
    def __init__(self, kind, detail=None):
        super().__init__(kind)
        self.kind, self.detail = kind, detail


class WeatherTool:
    def __init__(self, offline: bool = False, default_location: str = DEFAULT_LOCATION, timeout: float = 10.0):
        self.offline = offline
        self.default_location = default_location
        self.timeout = timeout
        self.gaz = gazetteer()
        self._forecasts = {}
        self._geo_path = CACHE_DIR / "geocode.json"
        self._geo = json.loads(self._geo_path.read_text(encoding="utf-8")) if self._geo_path.exists() else {}

    # ---------- places ----------
    def resolve_place(self, place):
        canon = self.default_location if place is None or place.get("canonical") == HOME else place.get("canonical")
        if canon and self.gaz.coords(canon):
            lat, lon = self.gaz.coords(canon)
            names = {l: self.gaz.display(canon, l) for l in ("en", "hi", "ta")}
            return {"canonical": canon, "lat": lat, "lon": lon, "names": names}
        return self._geocode(place["surface"])

    def _geocode(self, surface):
        if surface in self._geo:
            return self._geo[surface]
        if self.offline:
            raise WeatherError("place_not_found", surface)
        toks = tokenize(surface)
        queries = [" ".join(toks)]
        if toks and token_script(toks[-1]) == "ta":
            queries += [" ".join(toks[:-1] + [c]) for c in tamil_stem_candidates(toks[-1])[1:]]
        for q in queries:
            try:
                r = requests.get(GEOCODE_URL, params={"name": q, "count": 1, "language": "en"}, timeout=self.timeout)
                r.raise_for_status()
                res = r.json().get("results")
            except requests.RequestException as e:
                raise WeatherError("api", str(e))
            if res:
                g = res[0]
                out = {"canonical": g["name"], "lat": g["latitude"], "lon": g["longitude"], "names": {"en": g["name"], "hi": surface, "ta": surface}}
                self._geo[surface] = out
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                self._geo_path.write_text(json.dumps(self._geo, ensure_ascii=False, indent=1), encoding="utf-8")
                return out
        raise WeatherError("place_not_found", surface)

    # ---------- forecast ----------
    def _fetch(self, lat, lon):
        key = (round(lat, 3), round(lon, 3))
        hit = self._forecasts.get(key)
        if hit and time.time() - hit[0] < 600:
            return hit[1]
        data = self._mock(lat, lon) if self.offline else self._fetch_api(lat, lon)
        self._forecasts[key] = (time.time(), data)
        return data

    def _fetch_api(self, lat, lon):
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,wind_speed_10m_max",
            "hourly": "temperature_2m,precipitation_probability,weather_code",
            "timezone": "auto",
            "forecast_days": FORECAST_DAYS,
            "past_days": 1,
        }
        try:
            r = requests.get(FORECAST_URL, params=params, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            raise WeatherError("api", str(e))

    @staticmethod
    def _mock(lat, lon):
        """Deterministic fake forecast with the Open-Meteo schema, for offline tests and demos."""
        seed = int(hashlib.md5(f"{lat:.2f},{lon:.2f}".encode()).hexdigest(), 16)
        today = dt.date.today()
        days = [today + dt.timedelta(days=i) for i in range(-1, FORECAST_DAYS)]
        codes = [0, 2, 3, 61, 80, 95, 1, 63]
        daily = {"time": [], "weather_code": [], "temperature_2m_max": [], "temperature_2m_min": [], "precipitation_probability_max": [], "precipitation_sum": [], "wind_speed_10m_max": []}
        hourly = {"time": [], "temperature_2m": [], "precipitation_probability": [], "weather_code": []}
        for i, d in enumerate(days):
            s = (seed >> (i % 60)) & 0xFFFF
            code = codes[s % len(codes)]
            tmax = 24 + s % 12
            pop = 70 + s % 30 if code >= 61 else s % 30
            daily["time"].append(d.isoformat())
            daily["weather_code"].append(code)
            daily["temperature_2m_max"].append(float(tmax))
            daily["temperature_2m_min"].append(float(tmax - 6 - s % 4))
            daily["precipitation_probability_max"].append(pop)
            daily["precipitation_sum"].append(round(pop / 20, 1) if code >= 61 else 0.0)
            daily["wind_speed_10m_max"].append(float(8 + s % 20))
            for h in range(24):
                hourly["time"].append(f"{d.isoformat()}T{h:02d}:00")
                hourly["temperature_2m"].append(float(tmax - 6 + 6 * (1 - abs(h - 14) / 14)))
                hourly["precipitation_probability"].append(pop)
                hourly["weather_code"].append(code)
        return {
            "timezone": "Asia/Kolkata",
            "current": {"time": f"{today.isoformat()}T12:00", "temperature_2m": float(tmax - 2), "relative_humidity_2m": 60 + seed % 30, "weather_code": daily["weather_code"][1], "wind_speed_10m": 10.0, "precipitation": 0.0},
            "daily": daily,
            "hourly": hourly,
        }

    # ---------- query ----------
    def query(self, slots: dict) -> dict:
        place = self.resolve_place(slots.get("place"))
        data = self._fetch(place["lat"], place["lon"])
        daily = data["daily"]
        dates = [dt.date.fromisoformat(d) for d in daily["time"]]
        today = dt.date.fromisoformat(data["current"]["time"][:10])
        t0 = dates.index(today)

        def day(offset):
            i = t0 + offset
            if not 0 <= i < len(dates):
                raise WeatherError("out_of_range", offset)
            return {
                "date": dates[i].isoformat(),
                "offset": offset,
                "weekday": dates[i].weekday(),
                "code": daily["weather_code"][i],
                "tmax": round(daily["temperature_2m_max"][i]),
                "tmin": round(daily["temperature_2m_min"][i]),
                "pop": daily["precipitation_probability_max"][i] or 0,
                "precip": daily["precipitation_sum"][i] or 0.0,
                "wind": round(daily["wind_speed_10m_max"][i]),
            }

        date = slots.get("date") or {"current": True}
        tod = slots.get("timeofday")
        payload = {"place": place, "descriptor": slots.get("descriptor") or "general", "timeofday": tod, "days": [], "current": None, "part": None}

        if "range" in date or "weekend" in date:
            if "weekend" in date:
                sat = (5 - today.weekday()) % 7
                a, b = sat, sat + 1
            else:
                a, b = date["range"]
            b = min(b, len(dates) - t0 - 1)
            payload["kind"] = "range"
            payload["days"] = [day(o) for o in range(a, b + 1)]
            return payload

        if "weekday" in date:
            offset = (date["weekday"] - today.weekday()) % 7
        else:
            offset = date.get("offset", 0)
        payload["days"] = [day(offset)]

        if tod:
            hours = PARTS_OF_DAY[tod]
            target = dates[t0 + offset].isoformat()
            idx = [i for i, t in enumerate(data["hourly"]["time"]) if t.startswith(target) and int(t[11:13]) in hours]
            h = data["hourly"]
            payload["kind"] = "part"
            payload["part"] = {
                "temp": round(sum(h["temperature_2m"][i] for i in idx) / len(idx)),
                "pop": max((h["precipitation_probability"][i] or 0) for i in idx),
                "code": max(h["weather_code"][i] for i in idx),
            }
        elif date.get("current"):
            c = data["current"]
            payload["kind"] = "current"
            payload["current"] = {
                "temp": round(c["temperature_2m"]),
                "humidity": c["relative_humidity_2m"],
                "code": c["weather_code"],
                "wind": round(c["wind_speed_10m"]),
            }
        else:
            payload["kind"] = "day"
        return payload
