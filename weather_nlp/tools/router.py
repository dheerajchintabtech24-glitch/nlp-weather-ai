"""Routes a language-independent state to a tool; it never sees the surface language."""

from ..config import WEATHER_INTENT
from .weather import WeatherError, WeatherTool


class ToolRouter:
    def __init__(self, weather: WeatherTool):
        self.tools = {WEATHER_INTENT: weather.query}

    def route(self, state: dict) -> dict:
        tool = self.tools.get(state["intent"])
        if tool is None:
            return {"status": "unsupported", "intent": state["intent"]}
        try:
            return {"status": "ok", "tool": state["intent"], "payload": tool(state["slots"])}
        except WeatherError as e:
            return {"status": "error", "error": e.kind, "detail": e.detail}
