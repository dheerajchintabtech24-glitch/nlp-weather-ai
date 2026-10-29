"""Grounded response generation in English, Hindi or Tamil.

Every number and condition in a response is read from the tool payload; templates only supply the
language. Tamil templates avoid inflecting place names (X பகுதியில், X வானிலை) so that arbitrary
place names stay grammatical without a morphological generator.
"""

WMO_GROUPS = [
    ((0,), "clear"),
    ((1, 2), "partly_cloudy"),
    ((3,), "cloudy"),
    ((45, 48), "fog"),
    ((51, 53, 55, 56, 57), "drizzle"),
    ((61, 63, 65, 66, 67), "rain"),
    ((71, 73, 75, 77, 85, 86), "snow"),
    ((80, 81, 82), "showers"),
    ((95, 96, 99), "thunderstorm"),
]

CONDITIONS = {
    "en": {"clear": "clear skies", "partly_cloudy": "partly cloudy", "cloudy": "cloudy", "fog": "foggy", "drizzle": "drizzle", "rain": "rain", "snow": "snow", "showers": "rain showers", "thunderstorm": "thunderstorms"},
    "hi": {"clear": "साफ़ आसमान", "partly_cloudy": "आंशिक बादल", "cloudy": "बादल छाए हुए", "fog": "कोहरा", "drizzle": "बूंदाबांदी", "rain": "बारिश", "snow": "बर्फबारी", "showers": "बारिश की बौछारें", "thunderstorm": "आंधी-तूफ़ान"},
    "ta": {"clear": "தெளிவான வானம்", "partly_cloudy": "ஓரளவு மேகமூட்டம்", "cloudy": "மேகமூட்டம்", "fog": "மூடுபனி", "drizzle": "தூறல்", "rain": "மழை", "snow": "பனிப்பொழிவு", "showers": "மழைச்சாரல்", "thunderstorm": "இடியுடன் கூடிய மழை"},
}

WEEKDAYS = {
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "hi": ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"],
    "ta": ["திங்கட்கிழமை", "செவ்வாய்க்கிழமை", "புதன்கிழமை", "வியாழக்கிழமை", "வெள்ளிக்கிழமை", "சனிக்கிழமை", "ஞாயிற்றுக்கிழமை"],
}
RELATIVE_DAYS = {
    "en": {-1: "yesterday", 0: "today", 1: "tomorrow", 2: "the day after tomorrow"},
    "hi": {-1: "बीते कल", 0: "आज", 1: "कल", 2: "परसों"},
    "ta": {-1: "நேற்று", 0: "இன்று", 1: "நாளை", 2: "நாளை மறுநாள்"},
}
WEEKDAY_PREFIX = {"en": "on ", "hi": "", "ta": ""}
WEEKDAY_SUFFIX = {"en": "", "hi": " को", "ta": ""}
WHY = {
    "en": {"clear": "clear weather", "rain": "rain", "showers": "showers", "drizzle": "drizzle", "thunderstorm": "thunderstorms", "snow": "snow", "fog": "fog", "wind": "strong wind"},
    "hi": {"clear": "साफ़ मौसम", "rain": "बारिश", "showers": "बौछारें", "drizzle": "बूंदाबांदी", "thunderstorm": "आंधी-तूफ़ान", "snow": "बर्फबारी", "fog": "कोहरा", "wind": "तेज़ हवा"},
    "ta": {"clear": "தெளிவான வானிலை", "rain": "மழை", "showers": "மழைச்சாரல்", "drizzle": "தூறல்", "thunderstorm": "இடியுடன் கூடிய மழை", "snow": "பனி", "fog": "மூடுபனி", "wind": "பலத்த காற்று"},
}


def clock_phrase(minutes: int, lang: str) -> str:
    minutes = max(1, int(round(minutes)))
    hours, mins = divmod(minutes, 60)
    if lang == "hi":
        if hours and mins:
            return f"{hours} घंटे {mins} मिनट"
        return f"{hours} घंटे" if hours else f"{mins} मिनट"
    if lang == "ta":
        if hours and mins:
            return f"{hours} மணி {mins} நிமிடம்"
        return f"{hours} மணி" if hours else f"{mins} நிமிடம்"
    if hours and mins:
        unit = "hour" if hours == 1 else "hours"
        return f"{hours} {unit} {mins} minutes"
    if hours:
        return "1 hour" if hours == 1 else f"{hours} hours"
    return f"{mins} minutes"


PARTS = {
    "en": {"morning": "morning", "afternoon": "afternoon", "evening": "evening", "night": "night"},
    "hi": {"morning": "सुबह", "afternoon": "दोपहर", "evening": "शाम", "night": "रात"},
    "ta": {"morning": "காலை", "afternoon": "மதியம்", "evening": "மாலை", "night": "இரவு"},
}

T = {
    "en": {
        "day": "{Day} in {place}: {cond}. High {tmax}°C, low {tmin}°C, {pop}% chance of rain.",
        "current": "Right now in {place}: {cond}, {temp}°C, humidity {hum}%. Today: high {tmax}°C, low {tmin}°C, {pop}% chance of rain.",
        "part": "{Day} {part} in {place}: {cond}, around {temp}°C, {pop}% chance of rain.",
        "rain_yes": "Yes, rain is likely {day} in {place} ({pop}% chance). Take an umbrella.",
        "rain_maybe": "There is a slight chance of rain {day} in {place} ({pop}%).",
        "rain_no": "Rain is unlikely {day} in {place} ({pop}% chance).",
        "snow_yes": "Yes, snow is expected {day} in {place}.",
        "snow_no": "No snow is expected {day} in {place}.",
        "temp": "{Day} in {place} the temperature will be between {tmin}°C and {tmax}°C.",
        "cold": "It will be cold, carry a jacket.",
        "hot": "It will be hot, stay hydrated.",
        "wind": "Winds up to {wind} km/h.",
        "range_head": "Forecast for {place}:",
        "range_line": "{wd}: {cond}, {tmin}–{tmax}°C, rain {pop}%",
        "out_of_range": "I can only forecast up to 16 days ahead.",
        "place_not_found": "Sorry, I couldn't find the place '{name}'.",
        "api": "Sorry, I couldn't reach the weather service.",
        "unsupported": "Sorry, right now I can only help with weather questions.",
        "need_place": "Tell me which city you are going to, or ask about its weather first. You can also say where you are starting, for example “I'm in Delhi”.",
        "same_place": "{place} is both the start and the destination. Name a different starting city, for example “I'm in Delhi”.",
        "origin_set": "I'll measure trips from {place}. Ask how long it takes to reach the city we were just talking about.",
        "travel": "From {origin} to {place} is {km} km by road. Because of {why} ({pop}% chance of rain), it should take about {eta}. In clear weather it would take about {base}.",
        "travel_clear": "From {origin} to {place} is {km} km by road. The weather looks manageable ({cond}, {pop}% chance of rain), so it should take about {eta}.",
    },
    "hi": {
        "day": "{day} {place} में मौसम: {cond}। अधिकतम तापमान {tmax}°C, न्यूनतम {tmin}°C, बारिश की संभावना {pop}%।",
        "current": "अभी {place} में मौसम: {cond}, तापमान {temp}°C, नमी {hum}%। आज अधिकतम {tmax}°C और न्यूनतम {tmin}°C रहेगा, बारिश की संभावना {pop}%।",
        "part": "{day} {part} {place} में मौसम: {cond}, तापमान लगभग {temp}°C, बारिश की संभावना {pop}%।",
        "rain_yes": "हाँ, {day} {place} में बारिश होने की संभावना है ({pop}%)। छाता साथ रखें।",
        "rain_maybe": "{day} {place} में हल्की बारिश की थोड़ी संभावना है ({pop}%)।",
        "rain_no": "{day} {place} में बारिश की संभावना कम है ({pop}%)।",
        "snow_yes": "हाँ, {day} {place} में बर्फबारी की संभावना है।",
        "snow_no": "{day} {place} में बर्फबारी की संभावना नहीं है।",
        "temp": "{day} {place} में तापमान {tmin}°C से {tmax}°C के बीच रहेगा।",
        "cold": "ठंड रहेगी, जैकेट साथ रखें।",
        "hot": "गर्मी रहेगी, पानी पीते रहें।",
        "wind": "हवा की गति {wind} किमी/घंटा तक रहेगी।",
        "range_head": "{place} का मौसम पूर्वानुमान:",
        "range_line": "{wd}: {cond}, {tmin}–{tmax}°C, बारिश {pop}%",
        "out_of_range": "मैं केवल अगले 16 दिनों तक का पूर्वानुमान दे सकता हूँ।",
        "place_not_found": "माफ़ कीजिए, मुझे '{name}' नाम की जगह नहीं मिली।",
        "api": "माफ़ कीजिए, मौसम सेवा से संपर्क नहीं हो पाया।",
        "unsupported": "माफ़ कीजिए, अभी मैं केवल मौसम से जुड़े सवालों में मदद कर सकता हूँ।",
        "need_place": "किस शहर तक का समय बताऊँ? पहले उस शहर का मौसम पूछें, या दोनों शहर बताएं। शुरुआत के लिए “मैं दिल्ली में हूँ” कह सकते हैं।",
        "same_place": "{place} शुरू और मंज़िल दोनों है। दूसरा शहर बताएं, जैसे “मैं दिल्ली में हूँ”।",
        "origin_set": "अब सफ़र {place} से नापूँगा। जिस शहर की बात हो रही थी, वहाँ पहुँचने में कितना समय लगेगा, यह पूछें।",
        "travel": "{origin} से {place} तक सड़क मार्ग {km} किमी है। {why} की वजह से ({pop}% बारिश की संभावना) लगभग {eta} लगेंगे। साफ़ मौसम में लगभग {base} लगते।",
        "travel_clear": "{origin} से {place} तक सड़क मार्ग {km} किमी है। मौसम संभलने लायक है ({cond}, बारिश की संभावना {pop}%), इसलिए लगभग {eta} लगेंगे।",
    },
    "ta": {
        "day": "{day} {place} வானிலை: {cond}. அதிகபட்ச வெப்பநிலை {tmax}°C, குறைந்தபட்ச வெப்பநிலை {tmin}°C, மழைக்கான வாய்ப்பு {pop}%.",
        "current": "தற்போது {place} வானிலை: {cond}, வெப்பநிலை {temp}°C, ஈரப்பதம் {hum}%. இன்று அதிகபட்சம் {tmax}°C, குறைந்தபட்சம் {tmin}°C, மழைக்கான வாய்ப்பு {pop}%.",
        "part": "{day} {part} {place} வானிலை: {cond}, வெப்பநிலை சுமார் {temp}°C, மழைக்கான வாய்ப்பு {pop}%.",
        "rain_yes": "ஆம், {day} {place} பகுதியில் மழை பெய்ய வாய்ப்பு உள்ளது ({pop}%). குடை எடுத்துச் செல்லுங்கள்.",
        "rain_maybe": "{day} {place} பகுதியில் லேசான மழைக்கு சிறிய வாய்ப்பு உள்ளது ({pop}%).",
        "rain_no": "{day} {place} பகுதியில் மழைக்கான வாய்ப்பு குறைவு ({pop}%).",
        "snow_yes": "ஆம், {day} {place} பகுதியில் பனிப்பொழிவு எதிர்பார்க்கப்படுகிறது.",
        "snow_no": "{day} {place} பகுதியில் பனிப்பொழிவு எதிர்பார்க்கப்படவில்லை.",
        "temp": "{day} {place} பகுதியில் வெப்பநிலை {tmin}°C முதல் {tmax}°C வரை இருக்கும்.",
        "cold": "குளிராக இருக்கும், ஜாக்கெட் எடுத்துச் செல்லுங்கள்.",
        "hot": "வெயில் அதிகமாக இருக்கும், நிறைய தண்ணீர் குடியுங்கள்.",
        "wind": "காற்றின் வேகம் மணிக்கு {wind} கி.மீ வரை இருக்கும்.",
        "range_head": "{place} வானிலை முன்னறிவிப்பு:",
        "range_line": "{wd}: {cond}, {tmin}–{tmax}°C, மழை {pop}%",
        "out_of_range": "அடுத்த 16 நாட்களுக்கான முன்னறிவிப்பை மட்டுமே என்னால் தர முடியும்.",
        "place_not_found": "மன்னிக்கவும், '{name}' என்ற இடத்தைக் கண்டுபிடிக்க முடியவில்லை.",
        "api": "மன்னிக்கவும், வானிலை சேவையை அணுக முடியவில்லை.",
        "unsupported": "மன்னிக்கவும், தற்போது வானிலை தொடர்பான கேள்விகளுக்கு மட்டுமே என்னால் உதவ முடியும்.",
        "need_place": "எந்த நகரத்துக்கு நேரம் சொல்ல வேண்டும்? முதலில் அந்த நகரத்தின் வானிலையைக் கேளுங்கள், அல்லது இரண்டு நகரங்களையும் சொல்லுங்கள்.",
        "same_place": "{place} தொடக்கமும் சேருமிடமும் ஒன்றாக உள்ளது. வேறு தொடக்க நகரத்தைச் சொல்லுங்கள்.",
        "origin_set": "இனி பயணத்தை {place} இலிருந்து அளப்பேன். இப்போது பேசிய நகரத்துக்கு எவ்வளவு நேரம் ஆகும் என்று கேளுங்கள்.",
        "travel": "{origin} இலிருந்து {place} வரை சாலை வழியாக {km} கி.மீ. {why} காரணமாக ({pop}% மழை வாய்ப்பு) சுமார் {eta} ஆகும். தெளிவான வானிலையில் சுமார் {base} ஆகும்.",
        "travel_clear": "{origin} இலிருந்து {place} வரை சாலை வழியாக {km} கி.மீ. வானிலை சமாளிக்கக்கூடியது ({cond}, மழை வாய்ப்பு {pop}%), எனவே சுமார் {eta} ஆகும்.",
    },
}


def condition_group(code: int) -> str:
    for codes, name in WMO_GROUPS:
        if code in codes:
            return name
    return "cloudy"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


class Responder:
    def __init__(self, cold_below: int = 15, hot_above: int = 35):
        self.cold_below, self.hot_above = cold_below, hot_above

    def day_phrase(self, day: dict, lang: str) -> str:
        rel = RELATIVE_DAYS[lang].get(day["offset"])
        if rel:
            return rel
        return WEEKDAY_PREFIX[lang] + WEEKDAYS[lang][day["weekday"]] + WEEKDAY_SUFFIX[lang]

    def _place_name(self, place: dict, lang: str) -> str:
        if not place:
            return ""
        return (place.get("names") or {}).get(lang) or place.get("canonical") or place.get("surface") or ""

    def render(self, result: dict, lang: str) -> str:
        t = T[lang]
        if result["status"] == "unsupported":
            return t["unsupported"]
        if result["status"] == "error":
            return t.get(result["error"], t["api"]).format(name=result.get("detail") or "", place=result.get("detail") or "")
        p = result["payload"]
        if p.get("kind") == "origin":
            return t["origin_set"].format(place=self._place_name(p.get("place"), lang))
        if p.get("kind") == "same_place":
            return t["same_place"].format(place=self._place_name(p.get("place"), lang))
        if p.get("kind") == "travel":
            trip = p["trip"]
            day = p["days"][0]
            fields = {
                "origin": self._place_name(p.get("origin"), lang),
                "place": self._place_name(p.get("place"), lang),
                "km": trip["km"],
                "pop": day["pop"],
                "eta": clock_phrase(trip["minutes"], lang),
                "base": clock_phrase(trip["base_minutes"], lang),
                "why": WHY[lang].get(trip["reason"], trip["reason"]),
                "cond": CONDITIONS[lang][condition_group(day["code"])],
            }
            key = "travel_clear" if trip["factor"] <= 1.0 else "travel"
            return t[key].format(**fields)
        place = p["place"]["names"].get(lang) or p["place"]["canonical"]
        cond = lambda code: CONDITIONS[lang][condition_group(code)]  # noqa: E731

        if p["kind"] == "range":
            lines = [t["range_head"].format(place=place)]
            for d in p["days"]:
                wd = self.day_phrase(d, lang) if d["offset"] in (0, 1) else WEEKDAYS[lang][d["weekday"]]
                lines.append(t["range_line"].format(wd=_cap(wd), cond=cond(d["code"]), tmin=d["tmin"], tmax=d["tmax"], pop=d["pop"]))
            return "\n".join(lines)

        d = p["days"][0]
        day = self.day_phrase(d, lang)
        f = dict(place=place, day=day, Day=_cap(day), tmax=d["tmax"], tmin=d["tmin"], pop=d["pop"], wind=d["wind"])
        desc = p["descriptor"]
        pop = p["part"]["pop"] if p["part"] else d["pop"]

        if desc == "rain":
            key = "rain_yes" if pop >= 50 or d["precip"] >= 2 else "rain_maybe" if pop >= 20 else "rain_no"
            return t[key].format(**{**f, "pop": pop})
        if desc == "snow":
            return t["snow_yes" if condition_group(d["code"]) == "snow" else "snow_no"].format(**f)
        if desc in ("temperature", "hot", "cold"):
            out = t["temp"].format(**f)
            if d["tmin"] < self.cold_below:
                out += " " + t["cold"]
            elif d["tmax"] >= self.hot_above:
                out += " " + t["hot"]
            return out

        if p["kind"] == "part":
            q = p["part"]
            out = t["part"].format(**{**f, "part": PARTS[lang][p["timeofday"]], "cond": cond(q["code"]), "temp": q["temp"], "pop": q["pop"]})
        elif p["kind"] == "current":
            c = p["current"]
            out = t["current"].format(**{**f, "cond": cond(c["code"]), "temp": c["temp"], "hum": c["humidity"]})
        else:
            out = t["day"].format(**{**f, "cond": cond(d["code"])})
        if desc == "wind":
            out += " " + t["wind"].format(**f)
        return out
