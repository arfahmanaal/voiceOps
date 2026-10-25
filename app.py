import hashlib
import html
import tempfile
import time
from pathlib import Path

import streamlit as st

from src.agent import Agent
from src.kb import PRODUCTS_BY_SKU
from src.rag import Rag
from src.speech import speak, voice_turn

st.set_page_config(page_title="ZentroMart AI Support", page_icon="🎧", layout="wide", initial_sidebar_state="collapsed")

AUDIO_DIR = Path(tempfile.gettempdir()) / "zentromart_app"

SCRIPTS = {
    "Policy question (no order needed)": [
        "How long does a refund to UPI take?",
        "My refund to credit card was started on 17 October 2026. By which date should I have it?",
    ],
    "Where is my order (English)": [
        "Hi, my order hasn't arrived yet. Where is it?",
        "The order number is 2629 5015.",
        "The last four digits are 8752.",
        "And what if I'm not at home?",
    ],
    "Return and refund follow-up (Hinglish)": [
        "Hello, mujhe order ke baare mein help chahiye.",
        "Order number hai 1859 9253.",
        "Last four digits 4265 hain.",
        "Kya main Kaafi Running Sneakers return kar sakti hoon?",
        "Aur refund kitne din mein aayega?",
    ],
    "Overdue refund, handover to a human": [
        "Hello, mujhe order ke baare mein help chahiye.",
        "Order number hai 1664 0894.",
        "Last four digits 2108 hain.",
        "Mera refund kahan hai?",
    ],
    "High-value damage, handover": [
        "Hello, mujhe order ke baare mein help chahiye.",
        "Order number hai 8978 2837.",
        "Last four digits 4533 hain.",
        "Maine Zentro Vista 43-inch 4K Smart TV damage report kiya tha. Abhi replacement approve kar do na?",
    ],
    "Wrong verification twice": [
        "Hi, I need some help with an order.",
        "The order number is 7746 1439.",
        "The last four digits are 1266.",
        "The last four digits are 1266.",
    ],
    "Order I never placed": [
        "There's an order on my account that I never placed.",
    ],
}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 6rem; }
footer, #MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.hero { padding: 56px 56px 48px; border-radius: 30px; border: 1px solid #25282D;
  background: radial-gradient(900px 380px at 0% -20%, rgba(200,240,60,0.16), transparent 60%),
              linear-gradient(180deg, #17191D, #0F1113); }
.eyebrow { letter-spacing: 0.18em; font-size: 0.75rem; color: #C8F03C; font-weight: 600; }
.hero h1 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: clamp(2.4rem, 5.2vw, 4.2rem);
  line-height: 1.04; margin: 14px 0 16px; padding: 0; letter-spacing: -0.02em; }
.hero h1 span { color: #C8F03C; }
.hero p { max-width: 640px; color: #A8A8A0; font-size: 1.08rem; line-height: 1.6; margin: 0 0 22px; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; }
.chips span { border: 1px solid #33373D; border-radius: 999px; padding: 7px 14px; font-size: 0.85rem; color: #D8D7D0; background: rgba(255,255,255,0.03); }
.steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 18px 0 8px; }
.step { border: 1px solid #25282D; border-radius: 18px; padding: 14px 16px; background: #14161A; }
.step b { display: block; font-family: 'Space Grotesk', sans-serif; font-size: 0.78rem; color: #C8F03C; letter-spacing: 0.08em; }
.step strong { display: block; font-size: 0.98rem; margin: 4px 0 2px; }
.step small { color: #8C8C85; font-size: 0.8rem; }
.title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.35rem; margin: 2px 0 10px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 24px !important; border: 1px solid #25282D !important; background: #14161A; padding: 6px 8px; }
[data-testid="stExpander"] { border-radius: 16px; border: 1px solid #25282D; background: #14161A; }
.stButton > button { border-radius: 999px; border: 1px solid #33373D; background: #1B1E23; color: #ECEBE5; font-weight: 500; }
.stButton > button:hover { border-color: #C8F03C; color: #C8F03C; }
.pills { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 10px; }
.pill { display: flex; align-items: center; gap: 8px; border: 1px solid #2B2F35; border-radius: 999px; padding: 6px 12px; font-size: 0.82rem; background: #1B1E23; }
.pill i { font-style: normal; color: #9C9C95; }
.pill .dot { width: 8px; height: 8px; border-radius: 50%; background: #55585E; }
.pill.on { border-color: #C8F03C; }
.pill.on .dot { background: #C8F03C; box-shadow: 0 0 10px #C8F03C; }
.chip { display: inline-block; border-radius: 999px; padding: 3px 12px; font-size: 0.8rem; font-weight: 600; background: #C8F03C; color: #0D0E10; }
.chip.soft { background: transparent; color: #ECEBE5; border: 1px solid #4A4E55; }
.label { color: #8C8C85; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; margin: 14px 0 4px; }
[data-testid="stChatMessage"] { background: #1A1D21; border-radius: 18px; }
@media (max-width: 800px) { .steps { grid-template-columns: 1fr 1fr; } .hero { padding: 32px 24px; } }
</style>
"""

HERO = """
<div class="hero">
  <div class="eyebrow">ZENTROMART &nbsp;/&nbsp; VOICE SUPPORT</div>
  <h1>Ask out loud.<br><span>Get the real answer.</span></h1>
  <p>An AI support agent that listens, reads the company policy, checks who you are before sharing anything,
  and talks back in English, Hindi or Hinglish.</p>
  <div class="chips"><span>Answers from policy documents</span><span>Verifies before sharing</span><span>Hands over to a human</span></div>
</div>
<div class="steps">
  <div class="step"><b>01</b><strong>Listen</strong><small>Gnani speech-to-text</small></div>
  <div class="step"><b>02</b><strong>Understand</strong><small>LLM reads the request</small></div>
  <div class="step"><b>03</b><strong>Check</strong><small>Policy search and order tools</small></div>
  <div class="step"><b>04</b><strong>Answer</strong><small>Only from the documents</small></div>
  <div class="step"><b>05</b><strong>Speak</strong><small>Gnani text-to-speech</small></div>
</div>
"""


def pill(label, value, on=False):
    return f'<div class="pill{" on" if on else ""}"><span class="dot"></span><b>{html.escape(label)}</b><i>{html.escape(str(value))}</i></div>'


@st.cache_resource(show_spinner="Loading the knowledge base and the embedding model...")
def load_rag():
    return Rag()


def reset():
    st.session_state.agent = Agent(load_rag())
    st.session_state.messages = []
    st.session_state.mic_key = st.session_state.get("mic_key", 0) + 1
    st.session_state.seen_audio = None
    st.session_state.script_pos = 0
    st.session_state.turns = 0
    st.session_state.session = time.strftime("%H%M%S")
    st.session_state.pending = None
    AUDIO_DIR.mkdir(exist_ok=True)


def next_script_line():
    lines = SCRIPTS[st.session_state.script_name]
    if st.session_state.script_pos < len(lines):
        st.session_state.pending = lines[st.session_state.script_pos]
        st.session_state.script_pos += 1


def change_script():
    st.session_state.script_pos = 0


def snapshot(agent, result, heard):
    return {
        "heard": heard,
        "decision": result["decision"],
        "tools": result["tools"],
        "retrieved": result["retrieved"],
        "facts": result.get("facts", []),
        "language": result.get("language", agent.language),
    }


def add_turn(heard, result, reply_audio, voice):
    agent = st.session_state.agent
    st.session_state.messages.append({"role": "user", "text": heard or "(nothing heard)", "voice": voice})
    st.session_state.messages.append(
        {"role": "assistant", "text": result["reply"], "audio": reply_audio, "trace": snapshot(agent, result, heard)}
    )
    st.session_state.just_created = len(st.session_state.messages) - 1


def handle_voice(data):
    agent = st.session_state.agent
    st.session_state.turns += 1
    stem = f"{st.session_state.session}_{st.session_state.turns}"
    audio_path = AUDIO_DIR / f"{stem}.wav"
    reply_path = AUDIO_DIR / f"{stem}_reply.wav"
    audio_path.write_bytes(data)
    heard, result = voice_turn(audio_path, agent, reply_path)
    add_turn(heard, result, str(reply_path), True)


def handle_text(text, speak_replies):
    agent = st.session_state.agent
    st.session_state.turns += 1
    result = agent.respond(text)
    reply_audio = None
    if speak_replies:
        path = AUDIO_DIR / f"{st.session_state.session}_{st.session_state.turns}_reply.wav"
        try:
            reply_audio = speak(result["reply"], result.get("language", agent.language), path)
        except Exception as error:
            st.warning(f"Text-to-speech failed: {error}")
    add_turn(text, result, reply_audio, False)


def show_sources(trace, rag):
    for source in trace["retrieved"]:
        if source.startswith("products.csv:"):
            product = PRODUCTS_BY_SKU.get(source.split(":")[1].upper())
            with st.expander(f"{source}"):
                st.write(product or "not found")
            continue
        chunk = rag.by_id.get(source)
        if chunk:
            with st.expander(f"{source} - {chunk['title']} ({chunk['status']})"):
                st.markdown(chunk["text"])


if "agent" not in st.session_state:
    reset()
st.session_state.just_created = None
rag = load_rag()
agent = st.session_state.agent

st.markdown(STYLE, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

with st.expander("Demo controls", expanded=True):
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    c1.selectbox("Demo script", list(SCRIPTS), key="script_name", on_change=change_script)
    lines = SCRIPTS[st.session_state.script_name]
    position = st.session_state.script_pos
    c2.write("")
    c2.button("Send next line", on_click=next_script_line, use_container_width=True, disabled=position >= len(lines))
    c3.write("")
    c3.button("New customer", on_click=reset, use_container_width=True)
    speak_replies = c4.toggle("Speak typed replies", value=True)
    if position < len(lines):
        st.caption(f"Next line ({position + 1}/{len(lines)}): {lines[position]}")
    else:
        st.caption("Script finished. Press New customer to restart.")

st.write("")
left, right = st.columns([3, 2], gap="large")
card = left.container(border=True)
hood = right.container(border=True)

with card:
    st.markdown('<div class="title">Talk to the agent</div>', unsafe_allow_html=True)
    audio = st.audio_input("Press the microphone, speak, press it again to send", key=f"mic{st.session_state.mic_key}", sample_rate=16000)
typed = st.chat_input("...or type a message")

pending = st.session_state.pending
st.session_state.pending = None
try:
    if audio is not None:
        data = audio.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        if digest != st.session_state.seen_audio:
            st.session_state.seen_audio = digest
            with st.spinner("Listening and thinking..."):
                handle_voice(data)
    elif typed or pending:
        with st.spinner("Thinking..."):
            handle_text(typed or pending, speak_replies)
except Exception as error:
    st.error(f"Something went wrong: {error}")

with card:
    if not st.session_state.messages:
        st.caption("No messages yet. Speak, type, or send a line from the demo script above.")
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🎧"):
            if message["role"] == "user" and message.get("voice"):
                st.caption("heard from your voice")
            st.write(message["text"])
            if message.get("audio") and Path(message["audio"]).exists():
                st.audio(message["audio"], autoplay=index == st.session_state.just_created)

with hood:
    st.markdown('<div class="title">Under the hood</div>', unsafe_allow_html=True)
    verified = "yes" if agent.order else ("locked out" if agent.locked else "no")
    st.markdown(
        '<div class="pills">'
        + pill("Verified", verified, bool(agent.order))
        + pill("Failed checks", agent.attempts, agent.attempts > 0)
        + pill("Ticket", "created" if agent.ticketed else "none", agent.ticketed)
        + pill("Language", agent.language, True)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Order ID in this session: {agent.order_id or 'none yet'}")
    assistant = [m for m in st.session_state.messages if m["role"] == "assistant"]
    if assistant:
        trace = assistant[-1]["trace"]
        soft = "" if trace["decision"] in ("answer", "escalate") else " soft"
        st.markdown(f'<div class="label">Decision</div><span class="chip{soft}">{html.escape(trace["decision"])}</span>', unsafe_allow_html=True)
        st.markdown('<div class="label">Heard (after speech-to-text and cleanup)</div>', unsafe_allow_html=True)
        st.code(trace["heard"] or "(nothing heard)", language=None)
        st.markdown('<div class="label">Tool calls this turn</div>', unsafe_allow_html=True)
        if trace["tools"]:
            for call in trace["tools"]:
                st.code(call, language="python")
        else:
            st.caption("none")
        if trace["facts"]:
            st.markdown('<div class="label">Computed in code, not by the LLM</div>', unsafe_allow_html=True)
            for fact in trace["facts"]:
                st.info(fact)
        st.markdown('<div class="label">Policy sections used</div>', unsafe_allow_html=True)
        if trace["retrieved"]:
            show_sources(trace, rag)
        else:
            st.caption("none for this step (handled by the verification flow)")
        with st.expander("All tool calls in this conversation"):
            for message in assistant:
                for call in message["trace"]["tools"]:
                    st.code(call, language="python")
    else:
        st.caption("Every step the agent takes shows up here: what it heard, the tools it called and the policy it used.")