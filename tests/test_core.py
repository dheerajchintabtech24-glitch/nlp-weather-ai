import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weather_nlp.data.massive import parse_annot_utt, to_annot_utt  # noqa: E402
from weather_nlp.dialect import DialectAugmenter, LexiconNormalizer  # noqa: E402
from weather_nlp.dialogue import Canonicalizer, ContextManager  # noqa: E402
from weather_nlp.knowledge import gazetteer  # noqa: E402
from weather_nlp.nlg import Responder  # noqa: E402
from weather_nlp.nlu.metrics import bio_spans, span_f1  # noqa: E402
from weather_nlp.nlu.predictor import RuleNLU  # noqa: E402
from weather_nlp.preprocessing import detect_language, fold, tamil_stem_candidates, tokenize  # noqa: E402
from weather_nlp.speech_text import dialectize, prepare_speech, to_speech_text  # noqa: E402
from weather_nlp.assistant import WeatherAssistant  # noqa: E402
from weather_nlp.tools import ToolRouter, WeatherTool  # noqa: E402


def test_bio_roundtrip():
    annot = "मुझे [date : इस सप्ताह] का मौसम बताओ"
    toks, tags = parse_annot_utt(annot)
    assert tags == ["O", "B-date", "I-date", "O", "O", "O"]
    assert to_annot_utt(toks, tags) == annot


def test_span_f1():
    gold = [["B-date", "O", "B-place_name", "I-place_name"]]
    assert span_f1(gold, gold)[2] == 1.0
    assert bio_spans(["I-date", "I-date"]) == {("date", 0, 2)}


def test_tokenize_and_fold():
    assert tokenize("What's the weather?") == ["what's", "the", "weather"]
    assert fold("बर्फ़") == fold("बर्फ")
    assert fold("हूँ") == fold("हूं")


def test_langid():
    assert detect_language("चेन्नई में मौसम")[0] == "hi"
    assert detect_language("சென்னை வானிலை")[0] == "ta"
    assert detect_language("Delli mein mosam kesa hai?")[0] == "hi"
    assert detect_language("what's the weather in Delhi tomorrow")[0] == "en"


def test_tamil_stemming_resolves_places():
    gaz = gazetteer()
    for surface, canon in [("சென்னையில்", "Chennai"), ("பெங்களூரில்", "Bengaluru"), ("திருவனந்தபுரத்தில்", "Thiruvananthapuram"),
                           ("கேரளாவில்", None), ("சென்னைல", "Chennai"), ("மதுரைக்கு", "Madurai"), ("கோவைல", "Coimbatore")]:
        assert gaz.lookup(surface) == canon, surface
    assert "சென்னை" in tamil_stem_candidates("சென்னையில்")


def test_augmentation_keeps_alignment():
    aug = DialectAugmenter(p_replace=1.0, p_particle=0.0)
    toks, tags = parse_annot_utt("[place_name : சென்னையில்] [date : நாளை] மழை பெய்யுமா")
    rec = {"tokens": toks, "tags": tags, "lang": "ta"}
    new, new_tags, changes = aug.transform(toks, tags, "ta", "kongu", random.Random(0))
    assert changes > 0 and len(new) == len(new_tags)
    assert new[0] == "சென்னைல" and new_tags[0] == "B-place_name"
    assert aug.variants(rec, "chennai", k=1)


def test_normalizer_is_dialect_gated():
    norm = LexiconNormalizer()
    std = tokenize("आज का मौसम कैसा है")
    assert norm.normalize(std, "hi", "standard") == std
    assert norm.normalize(tokenize("आज मौसम कइसन बा"), "hi", "bhojpuri") == tokenize("आज मौसम कैसा है")
    assert norm.normalize(tokenize("சென்னைல மழை வருமுங்களா சாமி"), "ta", "kongu")[0] == "சென்னையில்"


def test_canonical_dates_and_tense():
    canon = Canonicalizer()
    nlu = {"slots": [{"type": "date", "value": "कल"}]}
    assert canon.canonicalize(nlu, tokenize("कल बारिश होगी"))["date"] == {"offset": 1}
    assert canon.canonicalize(nlu, tokenize("कल बारिश हुई थी"))["date"] == {"offset": -1}
    assert canon.canonicalize({"slots": [{"type": "date", "value": "நாளைக்கு"}]}, ["நாளைக்கு"])["date"] == {"offset": 1}


def test_cross_lingual_carry_over():
    nlu, canon, cm = RuleNLU(), Canonicalizer(), ContextManager()
    for text in ["बेंगलुरु में कल बारिश होगी क्या", "சென்னையில்?"]:
        toks = tokenize(text)
        p = nlu.predict(toks)
        state, mode = cm.update(text, toks, detect_language(text)[0], p, canon.canonicalize(p, toks))
    assert mode == "followup"
    assert state["intent"] == "weather_query"
    assert state["slots"]["place"]["canonical"] == "Chennai"
    assert state["slots"]["date"] == {"offset": 1} and state["slots"]["descriptor"] == "rain"


def test_grounded_responses_all_languages():
    router, resp = ToolRouter(WeatherTool(offline=True)), Responder()
    state = {"intent": "weather_query", "slots": {"place": {"canonical": "Chennai", "surface": "x"}, "date": {"offset": 1}, "timeofday": None, "descriptor": "rain"}}
    result = router.route(state)
    pop = result["payload"]["days"][0]["pop"]
    for lang in ("en", "hi", "ta"):
        text = resp.render(result, lang)
        assert f"{pop}%" in text
    assert "சென்னை" in resp.render(result, "ta")
    state["slots"]["date"] = {"range": [0, 6]}
    assert len(resp.render(router.route(state), "hi").splitlines()) == 8


def test_spoken_form_expands_numbers_in_each_language():
    en = to_speech_text("High 32°C, low 26–28°C, 80% chance, wind 12 km/h.", "en")
    assert "thirty-two" in en and "eighty" in en and "twelve" in en
    assert "°" not in en and "%" not in en and not any(ch.isdigit() for ch in en)
    hi = to_speech_text("अधिकतम 32°C, बारिश 80%.", "hi")
    assert "बत्तीस" in hi and "अस्सी" in hi and "डिग्री" in hi and not any(ch.isdigit() for ch in hi)
    ta = to_speech_text("அதிகபட்சம் 32°C, மழை 80%, காற்று 12 கி.மீ.", "ta")
    assert "முப்பத்தி இரண்டு" in ta and "எண்பது" in ta and "டிகிரி" in ta and not any(ch.isdigit() for ch in ta)
    assert to_speech_text("-5°C", "en").startswith("minus ")


def test_dialect_wording_is_gated_and_keeps_the_language_voice():
    assert "बरखा" in dialectize("कल बारिश होगी", "hi", "bhojpuri")
    assert "काल्ह" in dialectize("कल बारिश होगी", "hi", "bhojpuri")
    assert "இன்னிக்கு" in dialectize("இன்று மழை", "ta", "chennai")
    assert "இன்னைக்கு" in dialectize("இன்று மழை", "ta", "kongu")
    spoken = prepare_speech("कल 32°C बारिश 80%", "hi", "bhojpuri", 0.9)
    assert spoken["speech_lang"] == "hi" and spoken["speech_dialect"] == "bhojpuri"
    assert "बरखा" in spoken["speech_text"] and "बत्तीस" in spoken["speech_text"]
    held = prepare_speech("कल 32°C बारिश 80%", "hi", "bhojpuri", 0.1)
    assert held["speech_dialect"] == "standard" and "बारिश" in held["speech_text"]


def test_failed_place_does_not_steal_there_from_a_later_city():
    class _TagsUnknown:
        """Like the neural tagger: an unknown name is still a place slot, with no gazetteer hit."""

        def predict(self, tokens):
            out = RuleNLU().predict(tokens)
            if "gondor" in tokens and not any(s.get("type") == "place_name" for s in out["slots"]):
                out["slots"] = list(out["slots"]) + [{"type": "place_name", "value": "gondor"}]
                out["intent"] = "weather_query"
                out["intent_conf"] = 0.9
            return out

    bot = WeatherAssistant(nlu=_TagsUnknown(), offline=True)
    miss = bot.handle("How is the weather in Gondor right now?")
    assert miss["tool"].get("status") == "error"
    vizag = bot.handle("What is the weather in Vizag right now?")
    assert vizag["tool"]["status"] == "ok"
    assert vizag["state"]["slots"]["place"]["canonical"] == "Visakhapatnam"
    there = bot.handle("How long to go there?")
    assert there["nlu"]["intent"] == "travel_time"
    assert there["tool"]["payload"]["kind"] == "travel"
    assert there["tool"]["payload"]["place"]["canonical"] == "Visakhapatnam"
    assert "gondor" not in (there.get("response") or "").lower()
    ap = WeatherAssistant(nlu=RuleNLU(), offline=True).handle("How is the weather in Andhra Pradesh right now?")
    assert ap["tool"]["status"] == "ok"
    assert ap["state"]["slots"]["place"]["canonical"] == "Andhra Pradesh"


def test_travel_followup_keeps_place_across_languages():
    bot = WeatherAssistant(nlu=RuleNLU(), offline=True)
    bot.handle("what's the weather in Chennai tomorrow")
    hi = bot.handle("वहाँ पहुँचने में कितना समय लगेगा?")
    assert hi["lang"] == "hi" and hi["nlu"]["intent"] == "travel_time"
    trip = hi["tool"]["payload"]
    assert trip["kind"] == "travel"
    assert trip["place"]["canonical"] == "Chennai"
    assert trip["origin"]["canonical"] == "Bengaluru"
    assert trip["trip"]["km"] > 50
    assert str(trip["trip"]["km"]) in hi["response"]
    assert hi["state"]["slots"]["place"]["canonical"] == "Chennai"
    ta = bot.handle("அங்க போக எவ்வளவு நேரம் ஆகும்?")
    assert ta["lang"] == "ta" and ta["tool"]["payload"]["place"]["canonical"] == "Chennai"
    bot.handle("i am in Delhi")
    routed = bot.handle("how long will it take to get there?")
    assert routed["tool"]["payload"]["origin"]["canonical"] == "Delhi"
    assert routed["tool"]["payload"]["place"]["canonical"] == "Chennai"
    bot.context.undo_last("how long will it take to get there?")
    assert bot.context.state.slots["place"]["canonical"] == "Chennai"
    vizag = bot.handle("How long to go to Vizag?")
    assert vizag["nlu"]["intent"] == "travel_time"
    assert vizag["tool"]["payload"]["place"]["canonical"] == "Visakhapatnam"
    assert vizag["tool"]["payload"]["kind"] == "travel"
    there = bot.handle("उदर जाने में कितना देर लगेगा")
    assert there["lang"] == "hi" and there["nlu"]["intent"] == "travel_time"
    assert there["tool"]["payload"]["place"]["canonical"] == "Visakhapatnam"


def test_speech_follows_question_language_not_reply_language():
    bot = WeatherAssistant(nlu=RuleNLU(), offline=True)
    out = bot.handle("बेंगलुरु में कल बारिश होगी क्या", reply_lang="en")
    assert out["lang"] == "hi" and out["speech_lang"] == "hi" and out["reply_lang"] == "en"
    assert not any(ch.isdigit() for ch in out["speech_text"])
    assert any(ch.isdigit() for ch in out["response"])
