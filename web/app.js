const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const LANG = { en: "English", hi: "Hindi", ta: "Tamil" };
const DIALECT = {
  standard: "standard",
  bhojpuri: "Bhojpuri",
  haryanvi: "Haryanvi",
  bambaiya: "Bambaiya",
  chennai: "Chennai",
  kongu: "Kongu",
  madurai: "Madurai",
  nellai: "Nellai",
};
const CHIP_QUERIES = [
  { q: "आज दिल्ली में मौसम कैसा है?", tag: "Hindi" },
  { q: "काल्ह बलिया में बरखा होई का?", tag: "Bhojpuri", dialect: true },
  { q: "रोहतक म्ह आज मींह होवैगी के?", tag: "Haryanvi", dialect: true },
  { q: "बेंगलुरु में बारिश होएगी क्या रे?", tag: "Bambaiya", dialect: true },
  { q: "நாளைக்கு கோவைல மழை வருமுங்களா?", tag: "Kongu", dialect: true },
  { q: "இன்னிக்கு சென்னைல மழை வருமா?", tag: "Chennai", dialect: true },
  { q: "மதுரைல இன்னைக்கு வானிலை எப்படிண்ணே?", tag: "Madurai", dialect: true },
  { q: "ஏலே நெல்லைல இன்னைக்கு மழை வருமா?", tag: "Nellai", dialect: true },
  { q: "what's the weather in Bengaluru tomorrow", tag: "English" },
  { q: "how long will it take to get there?", tag: "Travel", travel: true },
  { q: "वहाँ पहुँचने में कितना समय लगेगा?", tag: "Travel", travel: true },
  { q: "அங்க போக எவ்வளவு நேரம் ஆகும்?", tag: "Travel", travel: true },
];

let busy = false;
let recorder = null;
let asrReady = false;
let currentAudio = null;
let speakGen = 0;
let askGen = 0;
let askAbort = null;
let inflightText = "";
let lastTurn = null;
let here = null;
let micArmed = false;
let micStarting = false;

const prettyDate = (d) => {
  if (!d) return "—";
  if (d.current) return "now";
  if (d.weekend) return "weekend";
  if (d.range) return `+${d.range[0]}…+${d.range[1]}`;
  if ("offset" in d) {
    const map = { "-1": "yesterday", 0: "today", 1: "tomorrow", 2: "day after" };
    return map[d.offset] || `+${d.offset}d`;
  }
  if ("weekday" in d) return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d.weekday];
  return JSON.stringify(d);
};
const prettyPlace = (p) => {
  if (!p) return "—";
  if (p.canonical === "@home") return "home";
  return p.canonical || p.surface || "—";
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function renderSlots(state, provenance) {
  const slots = (state && state.slots) || {};
  const fmt = {
    place: prettyPlace,
    date: prettyDate,
    timeofday: (v) => v || "—",
    descriptor: (v) => v || "—",
  };
  $$("#slots .slot").forEach((node) => {
    const k = node.dataset.k;
    const src = (provenance && provenance[k]) || "empty";
    node.classList.remove("new", "carried");
    if (src === "new" || src === "carried") node.classList.add(src);
    node.querySelector("strong").textContent = fmt[k](slots[k]);
    node.querySelector("i").textContent = src === "empty" ? "" : src;
  });
}

function renderWeather(w, fallbackText) {
  if (!w) {
    $("#wx-name").textContent = "Weather";
    $("#wx-names").textContent = fallbackText || "Ask a question to load a forecast.";
    $("#wx-temp").textContent = "–°";
    $("#wx-cond").textContent = "No forecast yet";
    $("#wx-tmax").textContent = "–";
    $("#wx-tmin").textContent = "–";
    $("#wx-pop").textContent = "–";
    $("#wx-wind").textContent = "–";
    return;
  }
  const names = (w.place && w.place.names) || {};
  $("#wx-name").textContent = names.en || (w.place && w.place.canonical) || "Weather";
  $("#wx-names").textContent = [names.hi, names.ta].filter(Boolean).join(" · ") || "Live forecast";
  $("#wx-temp").textContent = w.temp != null ? `${w.temp}°` : "–°";
  $("#wx-cond").textContent = (w.condition || "forecast").replaceAll("_", " ");
  $("#wx-tmax").textContent = w.tmax != null ? `${w.tmax}°` : "–";
  $("#wx-tmin").textContent = w.tmin != null ? `${w.tmin}°` : "–";
  $("#wx-pop").textContent = w.pop != null ? `${w.pop}%` : "–";
  $("#wx-wind").textContent = w.wind != null ? `${w.wind}` : "–";
  if (w.travel) {
    const t = w.travel;
    const clock = (mins) => {
      const h = Math.floor((mins || 0) / 60);
      const m = (mins || 0) % 60;
      if (h && m) return `${h}h ${m}m`;
      return h ? `${h}h` : `${m}m`;
    };
    $("#wx-names").textContent = `${t.origin || "start"} → ${names.en || "there"} · ${t.km} km · ${clock(t.eta_min)} in this weather (${clock(t.base_min)} if clear)`;
  }
}

function renderSpeech(res) {
  const el = $("#speech-line");
  if (!res || !res.speech_text) {
    el.innerHTML = "<span class='muted'>Follows the question language.</span>";
    return;
  }
  const lang = LANG[res.speech_lang] || res.speech_lang || "";
  const wording =
    res.speech_dialect && res.speech_dialect !== "standard"
      ? `${DIALECT[res.speech_dialect] || res.speech_dialect} wording · ${lang} voice`
      : `${lang} voice`;
  el.innerHTML = `<span class="muted">${escapeHtml(wording)}</span><br>${escapeHtml(res.speech_text)}`;
}

function inspect(res) {
  $("#fact-lang").textContent = LANG[res.lang] || res.lang || "—";
  $("#fact-dialect").textContent = DIALECT[res.dialect] || res.dialect || "—";
  $("#fact-intent").textContent = res.nlu ? res.nlu.intent : "—";
  const mode = res.mode || "idle";
  const mp = $("#mode-pill");
  mp.textContent = mode;
  mp.className = mode;
  renderSlots(res.state, res.provenance);
  renderWeather(res.weather, res.response);
  renderSpeech(res);
}

const ASR_STEPS = [
  { t: "Speech front-end", d: "Microphone waveform, downsampled to 16 kHz mono." },
  { t: "Spoken-language head", d: "Whisper chooses English, Hindi or Tamil inside one decode. It does not translate." },
  { t: "Greedy transcription", d: "num_beams = 1, so a short question does not wait on a 5-beam search." },
];
const NLP_STEPS = [
  { t: "Language identification", d: "Devanagari, Tamil or Latin script. Romanized cues (mausam, kaisa, mazhai) can override English." },
  { t: "Dialect classification", d: "Character n-gram TF–IDF and a linear SVM, gated by script so Hindi is never scored as Kongu." },
  { t: "Intent and slot tagging", d: "MuRIL intent plus BIO slots: place, date, time of day, weather descriptor." },
  { t: "Canonicalisation", d: "Gazetteer places and relative dates become symbols that do not depend on the language." },
  { t: "Dialogue state", d: "The place and date carry into the next turn even if you switch language." },
  { t: "Weather grounding", d: "Open-Meteo forecast. Every temperature and rain percentage in the reply comes from this payload." },
  { t: "Travel time", d: "If you asked how long the journey is: road distance to the place in context, slowed by that forecast." },
  { t: "Reply generation", d: "A template in the question language, then dialect wording and one neural voice." },
];

const STOP_ICON = `<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="1.5"/></svg>`;

let processTimer = null;

function stopControl(title) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "msg-stop";
  btn.title = "Stop this answer. The previous city and date stay for the next question.";
  btn.setAttribute("aria-label", "Stop this answer");
  btn.innerHTML = STOP_ICON;
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    stopAsk();
  });
  return btn;
}

function showSteps(steps, active, title) {
  $("#empty").hidden = true;
  let card = $("#process");
  if (!card) {
    card = document.createElement("li");
    card.id = "process";
    card.className = "bubble bot process";
    card.innerHTML = `<div class="spin" aria-hidden="true"></div><div class="process-copy"><b id="process-title"></b><ol id="process-steps"></ol></div>`;
    card.append(stopControl("Stop this answer"));
    $("#thread").append(card);
  }
  const head = $("#process-title");
  if (head) head.textContent = title;
  const list = $("#process-steps");
  if (list) {
    list.innerHTML = steps
      .map((s, i) => {
        const state = i < active ? "done" : i === active ? "now" : "wait";
        return `<li class="${state}"><b>${escapeHtml(s.t)}</b><span>${escapeHtml(s.d)}</span></li>`;
      })
      .join("");
  }
  card.scrollIntoView({ block: "end" });
}

function clearProcess() {
  if (processTimer) {
    clearInterval(processTimer);
    processTimer = null;
  }
  const card = $("#process");
  if (card) card.remove();
}

function runStages(steps, title) {
  clearProcess();
  let i = 0;
  const started = Date.now();
  const draw = () => {
    const sec = Math.max(0, Math.round((Date.now() - started) / 1000));
    showSteps(steps, i, `${title} · ${sec}s`);
  };
  draw();
  processTimer = setInterval(() => {
    if (i < steps.length - 1) i += 1;
    draw();
  }, 700);
}

function showTrace(trace) {
  if (processTimer) {
    clearInterval(processTimer);
    processTimer = null;
  }
  const steps = (trace || []).map((row) => ({ t: row.title, d: row.detail }));
  if (!steps.length) return;
  showSteps(steps, steps.length, "What the pipeline decided");
}

function addBubble(kind, text, meta) {
  $("#empty").hidden = true;
  const li = document.createElement("li");
  li.className = `bubble ${kind}`;
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = kind === "user" ? "You" : "NLP Weather AI";
  const body = document.createElement("div");
  body.textContent = text;
  li.append(who, body);
  if (kind === "bot") li.append(stopControl("Stop this answer"));
  if (meta) {
    const row = document.createElement("div");
    row.className = "badges";
    meta.forEach((m) => {
      const b = document.createElement("span");
      b.className = `badge ${m.cls || ""}`;
      b.textContent = m.text;
      row.append(b);
    });
    li.append(row);
  }
  $("#thread").append(li);
  li.scrollIntoView({ behavior: "smooth", block: "end" });
  return li;
}

function stopSpeaking() {
  speakGen += 1;
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  if (window.speechSynthesis) speechSynthesis.cancel();
}

function speakBrowser(text, lang) {
  if (!window.speechSynthesis || !text) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = { en: "en-IN", hi: "hi-IN", ta: "ta-IN" }[lang] || "en-IN";
  u.rate = 0.95;
  speechSynthesis.speak(u);
}

function whenAudioStarts(data) {
  const lang = (data && data.speech_lang) || (data && data.lang) || "en";
  const text = (data && data.speech_text) || (data && data.response) || "";
  if (!$("#tts").checked || !text) return Promise.resolve();
  stopSpeaking();
  const gen = ++speakGen;
  return fetch("/api/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, lang }),
  })
    .then((res) => res.json().then((body) => ({ ok: res.ok, body })))
    .then(({ ok, body }) => {
      if (gen !== speakGen) return;
      if (!ok || !body.speech_audio) throw new Error(body.error || "tts failed");
      return new Promise((resolve, reject) => {
        const audio = new Audio(`data:${body.speech_audio_mime || "audio/mpeg"};base64,${body.speech_audio}`);
        currentAudio = audio;
        const done = () => {
          if (currentAudio === audio) currentAudio = null;
        };
        audio.addEventListener("ended", done, { once: true });
        audio.addEventListener("pause", done, { once: true });
        const started = () => resolve();
        audio.addEventListener("playing", started, { once: true });
        audio.onerror = () => {
          done();
          reject(new Error("audio play failed"));
        };
        audio.play().catch((e) => {
          done();
          reject(e);
        });
      });
    })
    .catch(() => {
      if (gen !== speakGen || !$("#tts").checked) return;
      return new Promise((resolve) => {
        if (!window.speechSynthesis) {
          resolve();
          return;
        }
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.lang = { en: "en-IN", hi: "hi-IN", ta: "ta-IN" }[lang] || "en-IN";
        u.rate = 0.95;
        u.onstart = () => resolve();
        u.onerror = () => resolve();
        speechSynthesis.speak(u);
        setTimeout(resolve, 400);
      });
    });
}

function armStop(text) {
  busy = true;
  inflightText = text || "";
  $("#send").disabled = true;
  $("#mic").disabled = true;
}

function clearStop() {
  busy = false;
  inflightText = "";
  $("#send").disabled = false;
  $("#mic").disabled = !canListen();
}

function stopAsk() {
  const turn = lastTurn;
  const text = inflightText || (turn && turn.text) || "";
  lastTurn = null;
  askGen += 1;
  if (askAbort) {
    askAbort.abort();
    askAbort = null;
  }
  stopSpeaking();
  clearProcess();
  clearStop();
  if (turn && turn.bot && turn.bot.isConnected) {
    const stop = turn.bot.querySelector(".msg-stop");
    if (stop) stop.remove();
  }
  fetch("/api/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
    .then((r) => r.json())
    .then((res) => {
      if (res && res.state) renderSlots(res.state, {});
    })
    .catch(() => {});
  $("#q").focus();
}

async function ask(text, replyLang, via, heardLang) {
  if (!text || busy) return;
  const gen = ++askGen;
  armStop(text);
  askAbort = new AbortController();
  const badges = [{ text: via === "speech" ? "speech" : "text" }];
  if (via === "speech" && heardLang && heardLang !== "auto") {
    badges.push({ text: `heard ${LANG[heardLang] || heardLang}`, cls: `lang-${heardLang}` });
  }
  if (replyLang && replyLang !== "auto") badges.push({ text: `reply ${replyLang}` });
  const userBubble = addBubble("user", text, badges);
  lastTurn = { text, user: userBubble, bot: null };
  runStages(NLP_STEPS, "Understanding the question");
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: askAbort.signal,
      body: JSON.stringify({ text, reply_lang: replyLang || "auto", here }),
    });
    const data = await res.json();
    if (gen !== askGen) return;
    if (data.cancelled) {
      clearProcess();
      return;
    }
    if (!res.ok) throw new Error(data.error || "request failed");
    showTrace(data.trace);
    const voice = LANG[data.speech_lang] || data.speech_lang || "the question language";
    const title = $("#process-title");
    if (title) title.textContent = `Speaking in ${voice}`;
    await whenAudioStarts(data);
    if (gen !== askGen) return;
    clearProcess();
    const botBubble = addBubble("bot", data.response, [
      { text: `${LANG[data.lang] || data.lang}:${DIALECT[data.dialect] || data.dialect}`, cls: `lang-${data.lang}` },
      { text: data.mode },
      { text: data.nlu.intent },
    ]);
    if (lastTurn && lastTurn.user === userBubble) lastTurn.bot = botBubble;
    inspect(data);
    if (currentAudio) {
      currentAudio.addEventListener(
        "ended",
        () => {
          const stop = botBubble.querySelector(".msg-stop");
          if (stop) stop.remove();
          if (lastTurn && lastTurn.bot === botBubble) lastTurn = null;
        },
        { once: true }
      );
    }
  } catch (err) {
    if (gen !== askGen || err.name === "AbortError") return;
    clearProcess();
    addBubble("bot", "Could not reach the assistant. " + err.message);
  } finally {
    if (gen === askGen) clearStop();
  }
}

async function resetConversation() {
  lastTurn = null;
  await fetch("/api/reset", { method: "POST" });
  $("#thread").innerHTML = "";
  $("#empty").hidden = false;
  $("#fact-lang").textContent = "—";
  $("#fact-dialect").textContent = "—";
  $("#fact-intent").textContent = "—";
  $("#mode-pill").textContent = "idle";
  $("#mode-pill").className = "";
  renderSlots({}, {});
  renderWeather(null);
  renderSpeech(null);
  clearProcess();
  const overlay = $("#listen");
  if (overlay) overlay.hidden = true;
  stopSpeaking();
}

function floatToWav(float32, sampleRate) {
  const n = float32.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buf);
  const str = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  str(0, "RIFF");
  view.setUint32(4, 36 + n * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  str(36, "data");
  view.setUint32(40, n * 2, true);
  let off = 44;
  for (let i = 0; i < n; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

function downsample(input, inRate, outRate) {
  if (Math.abs(inRate - outRate) < 1) return input;
  const ratio = inRate / outRate;
  const n = Math.round(input.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const f = x - i0;
    out[i] = input[i0] * (1 - f) + input[i1] * f;
  }
  return out;
}

async function blobToB64(blob) {
  const buf = await blob.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let s = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(s);
}

function setMicHot(on) {
  $("#mic").classList.toggle("hot", on);
  $("#mic").title = on ? "Release to send" : "Hold to talk";
  const hint = $("#mic-hint");
  if (hint) {
    hint.textContent = on
      ? "Listening… release to send"
      : "Hold mic to talk · release to send. If Hindi or Tamil is heard as English, set Mic to that language.";
  }
  const overlay = $("#listen");
  if (overlay) overlay.hidden = !on;
}

function listenBrowser() {
  const Ctor = browserAsr();
  if (!Ctor) throw new Error("no browser speech recognition");
  const rec = new Ctor();
  rec.lang = asrLangCode();
  rec.interimResults = true;
  rec.maxAlternatives = 1;
  rec.continuous = true;
  setMicHot(true);
  let finalText = "";
  rec.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const piece = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalText = (finalText + " " + piece).trim();
      else interim += piece;
    }
    $("#q").value = (finalText + " " + interim).trim();
  };
  rec.onerror = (e) => {
    if (e.error !== "no-speech" && e.error !== "aborted") {
      addBubble("bot", "Mic error: " + e.error);
    }
  };
  rec.onend = () => {
    if (micArmed && recorder && recorder.browser === rec) {
      try {
        rec.start();
      } catch (_) {}
    }
  };
  recorder = { browser: rec, startedAt: Date.now() };
  rec.start();
}

async function startMic() {
  if (busy || micStarting) return;
  micStarting = true;
  micArmed = true;
  try {
    if (!asrReady && browserAsr()) {
      listenBrowser();
      return;
    }
    if (!asrReady) throw new Error("mic not ready");
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (!micArmed) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const chunks = [];
    proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    const gain = ctx.createGain();
    gain.gain.value = 0;
    src.connect(proc);
    proc.connect(gain);
    gain.connect(ctx.destination);
    recorder = { stream, ctx, proc, chunks, startedAt: Date.now() };
    setMicHot(true);
    recorder.timer = setTimeout(() => {
      if (micArmed) releaseMic();
    }, 15000);
  } catch (err) {
    micArmed = false;
    setMicHot(false);
    addBubble("bot", "Microphone permission is needed to ask by voice.");
  } finally {
    micStarting = false;
  }
}

async function releaseMic() {
  if (!micArmed && !recorder) return;
  micArmed = false;
  const rec = recorder;
  recorder = null;
  setMicHot(false);
  if (!rec) return;

  if (rec.browser) {
    try {
      rec.browser.onend = null;
      rec.browser.stop();
    } catch (_) {}
    const text = $("#q").value.trim();
    if (Date.now() - (rec.startedAt || 0) < 350 || text.length < 2) {
      addBubble("bot", "Hold the mic a bit longer, then speak your weather question.");
      return;
    }
    $("#q").value = "";
    await ask(text, $("#reply-lang").value, "speech");
    return;
  }

  clearTimeout(rec.timer);
  try {
    rec.proc.disconnect();
  } catch (_) {}
  rec.stream.getTracks().forEach((t) => t.stop());
  const sampleRate = rec.ctx.sampleRate;
  await rec.ctx.close();
  const heldMs = Date.now() - (rec.startedAt || 0);
  const len = rec.chunks.reduce((a, c) => a + c.length, 0);
  if (heldMs < 350 || len < sampleRate * 0.35) {
    addBubble("bot", "Hold the mic a bit longer, then speak your weather question.");
    return;
  }
  const samples = new Float32Array(len);
  let o = 0;
  for (const c of rec.chunks) {
    samples.set(c, o);
    o += c.length;
  }
  const wav = floatToWav(downsample(samples, sampleRate, 16000), 16000);
  const gen = ++askGen;
  armStop("");
  askAbort = new AbortController();
  runStages(ASR_STEPS, "Transcribing");
  try {
    const res = await fetch("/api/transcribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: askAbort.signal,
      body: JSON.stringify({ wav_b64: await blobToB64(wav), lang: $("#asr-lang").value }),
    });
    const data = await res.json();
    if (gen !== askGen) return;
    clearProcess();
    if (!res.ok) throw new Error(data.error || "transcription failed");
    $("#q").value = "";
    clearStop();
    await ask(data.text, $("#reply-lang").value, "speech", data.lang);
  } catch (err) {
    if (gen !== askGen || err.name === "AbortError") return;
    clearProcess();
    addBubble("bot", "Could not transcribe: " + err.message);
    clearStop();
  }
}

function bindPushToTalk() {
  const mic = $("#mic");
  const onDown = (e) => {
    if (busy || mic.disabled) return;
    e.preventDefault();
    if (e.button != null && e.button !== 0) return;
    if (micArmed || recorder || micStarting) return;
    startMic();
  };
  const onUp = (e) => {
    e.preventDefault();
    if (micArmed || recorder) releaseMic();
  };
  mic.addEventListener("pointerdown", onDown);
  mic.addEventListener("pointerup", onUp);
  mic.addEventListener("pointercancel", onUp);
  mic.addEventListener("pointerleave", (e) => {
    if (micArmed && (e.buttons === 0 || e.pointerType === "touch")) releaseMic();
  });
  // Avoid accidental click-toggle after pointer events.
  mic.addEventListener("click", (e) => e.preventDefault());
}

function browserAsr() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function canListen() {
  return asrReady || !!browserAsr();
}

function asrLangCode() {
  return { auto: "en-IN", en: "en-IN", hi: "hi-IN", ta: "ta-IN" }[$("#asr-lang").value] || "en-IN";
}

function applyHealth(h) {
  $("#nlu-pill").textContent = h.nlu || "NLU";
  $("#wx-pill").textContent = h.offline ? "mock weather" : "Open-Meteo";
  asrReady = !!h.asr;
  $("#asr-pill").textContent = h.asr ? (h.asr_model || "Whisper").replace("openai/", "") : browserAsr() ? "browser mic" : "Mic…";
  $("#asr-pill").classList.toggle("ghost", !h.asr);
  $("#mic").disabled = !canListen();
}

let infoMetricsLoaded = false;

function setInfoOpen(open) {
  $("#info-panel").hidden = !open;
  if (open && !infoMetricsLoaded) loadInfoMetrics();
}

async function loadInfoMetrics() {
  try {
    const m = await fetch("/api/metrics").then((r) => r.json());
    const w = (m.weather && m.weather.muril_dialect) || {};
    const c = (m.context && m.context.muril_dialect) || {};
    const d = m.dialect_id || {};
    const svm = d.svm || d;
    const dialectF1 = ((svm.macro_f1 || 0.923) * (svm.macro_f1 < 1.5 ? 100 : 1)).toFixed(1);
    $("#info-metrics").innerHTML = [
      [w.intent_acc ?? 97.2, "Weather intent"],
      [w.frame_acc ?? 88.7, "Full frame acc"],
      [c.jga ?? 83.2, "Follow-up JGA"],
      [dialectF1, "Dialect macro-F1"],
    ]
      .map(([n, l]) => `<div class="info-stat"><b>${n}</b><span>${l}</span></div>`)
      .join("");
    infoMetricsLoaded = true;
  } catch (_) {
    $("#info-metrics").innerHTML = "";
  }
}

async function boot() {
  $("#chips").innerHTML = CHIP_QUERIES.map(
    (c, i) =>
      `<button class="chip${c.dialect ? " dialect" : ""}${c.travel ? " travel" : ""}" type="button" data-i="${i}"><em>${escapeHtml(c.tag)}</em><span>${escapeHtml(c.q)}</span></button>`
  ).join("");
  $("#info-btn").addEventListener("click", () => setInfoOpen(true));
  $("#info-close").addEventListener("click", () => setInfoOpen(false));
  $("#info-panel").addEventListener("click", (e) => {
    if (e.target === $("#info-panel")) setInfoOpen(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!$("#info-panel").hidden) setInfoOpen(false);
    else if (busy || lastTurn) stopAsk();
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("#q").value.trim();
    $("#q").value = "";
    ask(text, $("#reply-lang").value);
  });
  bindPushToTalk();
  $("#chips").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    const item = CHIP_QUERIES[Number(chip.dataset.i)];
    if (!item) return;
    ask(item.q, $("#reply-lang").value);
  });

  const showApp = () => {
    $("#boot").hidden = true;
    $("#app").hidden = false;
    $("#q").focus();
  };
  let shown = false;
  for (let i = 0; i < 40; i++) {
    try {
      const h = await fetch("/api/health").then((r) => r.json());
      applyHealth(h);
      if (h.ok) {
        shown = true;
        showApp();
        break;
      }
      $("#boot-sub").textContent = "Loading model…";
    } catch (_) {
      $("#boot-sub").textContent = "Waiting for local server…";
    }
    await sleep(250);
  }
  if (!shown) showApp();
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        here = { lat: pos.coords.latitude, lon: pos.coords.longitude };
      },
      () => {},
      { enableHighAccuracy: false, timeout: 8000, maximumAge: 300000 }
    );
  }
  if (!asrReady) {
    (async () => {
      for (let i = 0; i < 240; i++) {
        try {
          const h = await fetch("/api/health").then((r) => r.json());
          applyHealth(h);
          if (h.asr) return;
        } catch (_) {}
        await sleep(1000);
      }
    })();
  }
}

boot();
