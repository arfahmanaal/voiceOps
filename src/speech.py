import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

from src.kb import PRODUCTS
from src.textnorm import normalize_speech

STT_LANGUAGE = os.getenv("GNANI_STT_LANGUAGE", "en-IN")
STT_FORMAT = os.getenv("GNANI_STT_FORMAT", "transcribe")
TTS_MODEL = os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")
HINDI_VOICE = os.getenv("GNANI_VOICE_HI", "Nalini")
VOICES = {
    "en": (os.getenv("GNANI_VOICE_EN", "Kaveri"), "en-IN"),
    "hi": (HINDI_VOICE, "hi-IN"),
    "hi-en": (os.getenv("GNANI_VOICE_HINGLISH", HINDI_VOICE), "hi-IN"),
}
V3_VOICE = os.getenv("GNANI_VOICE_V3", "Kalpana")
RUPEES = {"en": "rupees", "hi": "रुपये", "hi-en": "rupaye"}
UNCLEAR = {
    "en": "Sorry, I didn't catch that. Could you please say it again?",
    "hi": "माफ़ कीजिए, मुझे सुनाई नहीं दिया। क्या आप दोबारा बोल सकते हैं?",
    "hi-en": "Sorry, mujhe sunai nahi diya. Kya aap dobara bol sakte hain?",
}
TROUBLE = {
    "en": "Sorry, I am having trouble right now. A human colleague can help you; please try again shortly.",
    "hi": "माफ़ कीजिए, अभी मुझे दिक्कत आ रही है। कृपया थोड़ी देर बाद फिर कोशिश करें।",
    "hi-en": "Sorry, abhi mujhe dikkat aa rahi hai. Kripya thodi der baad phir koshish karein.",
}
CACHE = Path(".cache") / "tts"
DIGIT_WORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine)"
PREFIXES = "|".join(sorted({product["sku"].split("-")[0].lower() for product in PRODUCTS}))
SKU_SPLIT = re.compile(r"\b([a-z]{3})\s*-\s*(?=" + DIGIT_WORD + r"\b)", re.I)
SKU_JOIN = re.compile(r"\b(" + PREFIXES + r")[\s\-]*(\d{4})\b", re.I)

_stt = None
_tts = None


def clean_transcript(text):
    text = re.sub(r"\bu\s+p\s+i\b", "UPI", text, flags=re.I)
    text = SKU_SPLIT.sub(r"\1 ", text)
    text = normalize_speech(text)
    return SKU_JOIN.sub(lambda match: f"{match.group(1).upper()}-{match.group(2)}", text).strip()


def speakable(text, language="en"):
    rupees = RUPEES.get(language, "rupees")
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"₹\s*([\d,]+)", rf"\1 {rupees}", text)
    text = re.sub(r"\bINR\s*([\d,]+)", rf"\1 {rupees}", text)
    text = re.sub(r"\b\d{6,}\b", lambda match: " ".join(match.group()[i : i + 4] for i in range(0, len(match.group()), 4)), text)
    return re.sub(r"\s+", " ", text).strip()


def stt_client():
    global _stt
    if _stt is None:
        from gnani.stt import GnaniSTTClient

        _stt = GnaniSTTClient()
    return _stt


def tts_client():
    global _tts
    if _tts is None:
        from gnani.tts import GnaniTTSClient

        _tts = GnaniTTSClient()
    return _tts


def listen(audio_path, language=None):
    language = language or STT_LANGUAGE
    formats = [STT_FORMAT] if STT_FORMAT == "verbatim" else [STT_FORMAT, "verbatim"]
    for audio_format in formats:
        try:
            result = stt_client().transcribe(str(audio_path), language_code=language, format=audio_format)
            return clean_transcript(result.get("transcript", "") or "")
        except Exception as error:
            print(f"[stt error] {audio_path} ({audio_format}): {error}", file=sys.stderr, flush=True)
    return ""


def speak(text, language, out_path):
    spoken = speakable(text, language)
    if TTS_MODEL == "timbre-v2.5":
        voice, code = VOICES.get(language, VOICES["en"])
    else:
        voice, code = V3_VOICE, None
    key = hashlib.sha256(f"{TTS_MODEL}|{voice}|{code}|{spoken}".encode("utf-8")).hexdigest()[:24]
    cached = CACHE / f"{key}.wav"
    if not cached.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        options = {"language": code} if code else {}
        tts_client().synthesize(spoken, voice=voice, model=TTS_MODEL, output_file=str(cached), **options)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, destination)
    return str(destination)


def voice_turn(audio_path, agent, reply_path):
    heard = listen(audio_path)
    if not heard.strip():
        language = agent.language
        result = {"reply": UNCLEAR.get(language, UNCLEAR["en"]), "decision": "clarify", "tools": [], "retrieved": [], "language": language}
    else:
        result = agent.respond(heard)
    speak(result["reply"], result.get("language", agent.language), reply_path)
    return heard, result