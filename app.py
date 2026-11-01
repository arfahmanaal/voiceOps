import html
import inspect
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent import Agent
from src.kb import PRODUCTS_BY_SKU
from src.rag import Rag
from src.speech import speak, voice_turn

st.set_page_config(page_title="ZentroMart AI Support Assistant", page_icon=":material/support_agent:", layout="wide", initial_sidebar_state="collapsed")

AUDIO_DIR = Path("demo_audio")

SCENARIOS = {
    "Policy question": [
        "How long does a refund to UPI take?",
        "My refund to credit card was started on 17 October 2026. By which date should I have it?",
    ],
    "Order status": [
        "Hi, my order hasn't arrived yet. Where is it?",
        "The order number is 2629 5015.",
        "The last four digits are 8752.",
        "And what if I'm not at home?",
    ],
    "Return and refund (Hinglish)": [
        "Hello, mujhe order ke baare mein help chahiye.",
        "Order number hai 1859 9253.",
        "Last four digits 4265 hain.",
        "Kya main Kaafi Running Sneakers return kar sakti hoon?",
        "Aur refund kitne din mein aayega?",
    ],
    "Overdue refund": [
        "Hello, mujhe order ke baare mein help chahiye.",
        "Order number hai 1664 0894.",
        "Last four digits 2108 hain.",
        "Mera refund kahan hai?",
    ],
    "High-value damage": [
        "Hello, mujhe order ke baare mein help chahiye.",
        "Order number hai 8978 2837.",
        "Last four digits 4533 hain.",
        "Maine Zentro Vista 43-inch 4K Smart TV damage report kiya tha. Abhi replacement approve kar do na?",
    ],
    "Failed verification": [
        "Hi, I need some help with an order.",
        "The order number is 7746 1439.",
        "The last four digits are 1266.",
        "The last four digits are 1266.",
    ],
    "Unrecognised order": [
        "There's an order on my account that I never placed.",
    ],
}

DECISIONS = {
    "answer": "Answered",
    "clarify": "Asked for more details",
    "refuse": "Declined under policy",
    "escalate": "Handed over to a colleague",
    "error": "Technical problem",
}
LANGUAGES = {"en": "English", "hi": "Hindi", "hi-en": "Hinglish"}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { --accent: light-dark(#0A7A63, #4FE3B5); --on-accent: light-dark(#FFFFFF, #06231B);
  --line: color-mix(in srgb, currentColor 15%, transparent);
  --tint: color-mix(in srgb, currentColor 5%, transparent);
  --muted: color-mix(in srgb, currentColor 60%, transparent);
  --raised: light-dark(#FFFFFF, #111B18);
  --panel-h: max(440px, calc(100vh - 6rem));
  font-family: 'Inter', system-ui, sans-serif; }
html, body, .stApp, [data-testid="stMain"], [data-testid="stAppViewContainer"] { overflow: hidden !important; }
.block-container { max-width: 1500px; padding: 0.8rem 1.6rem 0.4rem; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
header[data-testid="stHeader"], header[data-testid="stHeader"] * { pointer-events: none !important; }
header[data-testid="stHeader"] [data-testid="stMainMenu"], header[data-testid="stHeader"] [data-testid="stMainMenu"] * { pointer-events: auto !important; }
.st-key-topbar { padding-right: 3rem; }
.st-key-topbar [data-testid="stMarkdown"], .st-key-topbar [data-testid="stMarkdownContainer"] { margin: 0 !important; }
.st-key-topbar [data-testid="stElementContainer"] { margin: 0 !important; }
[data-testid="stStatusWidget"] { display: none !important; }
[data-stale="true"] { opacity: 1 !important; transition: none !important; }
.stApp::before { content: ""; position: fixed; top: 0; left: 0; height: 3px; width: 100%; z-index: 1000000; opacity: 0;
  background: linear-gradient(90deg, transparent, var(--accent), transparent); transform: translateX(-100%); }
[data-testid="stPopoverBody"] { max-width: 540px; }
.stApp[data-test-script-state="running"]::before { opacity: 1; animation: slide 1.1s ease-in-out infinite; }
@keyframes slide { from { transform: translateX(-100%); } to { transform: translateX(100%); } }
.brand { display: flex; align-items: center; gap: 13px; }
.brand .mark { width: 44px; height: 44px; border-radius: 13px; background: var(--accent); color: var(--on-accent);
  display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.35rem; }
.brand .name { font-weight: 800; font-size: 1.5rem; letter-spacing: -0.025em; line-height: 1.05; }
.brand .sub { font-size: 0.68rem; color: var(--accent); font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; margin-top: 3px; }
.panel-title { display: flex; align-items: center; gap: 9px; font-size: 1.12rem; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 0.65rem; }
.panel-title small { margin-left: auto; font-size: 0.74rem; font-weight: 500; color: var(--muted); letter-spacing: 0; }
.label { font-size: 0.64rem; letter-spacing: 0.14em; font-weight: 600; text-transform: uppercase; color: var(--muted); margin: 1.15rem 0 0.45rem; }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 9px 14px; }
.kv .k { font-size: 0.78rem; color: var(--muted); }
.kv .v { text-align: right; font-size: 0.9rem; font-weight: 600; }
.kv .v.on { color: var(--accent); font-weight: 700; }
.chip { display: inline-block; padding: 4px 13px; border-radius: 999px; font-size: 0.82rem; font-weight: 700; background: var(--accent); color: var(--on-accent); }
.chip.neutral { background: transparent; color: inherit; border: 1px solid var(--line); }
.quote { border-left: 3px solid var(--accent); padding: 6px 0 6px 11px; font-size: 0.88rem; font-style: italic; background: var(--tint); border-radius: 0 10px 10px 0; }
.action { padding: 8px 11px; border: 1px solid var(--line); border-radius: 10px; font-size: 0.84rem; margin-bottom: 6px; background: var(--tint); }
.action::before { content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--accent); margin-right: 9px; }
.next { padding: 10px 12px; border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 12px; font-size: 0.92rem; background: var(--tint); margin: 6px 0 12px; }
.next small { display: block; font-size: 0.64rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 3px; }
.empty { text-align: center; padding: 4rem 1.2rem; color: var(--muted); font-size: 0.9rem; line-height: 1.55; }
.empty b { display: block; color: inherit; font-size: 1.15rem; font-weight: 700; margin-bottom: 6px; }
.stButton > button, [data-testid="stPopover"] > div > button { border-radius: 10px; font-weight: 600; font-size: 0.86rem; }
.stButton > button:hover, [data-testid="stPopover"] > div > button:hover { border-color: var(--accent); color: var(--accent); }
.st-key-left_panel, .st-key-right_panel { background: var(--tint) !important; }
.st-key-center_panel { background: var(--raised) !important; box-shadow: 0 1px 2px rgba(0,0,0,0.05), 0 10px 28px rgba(0,0,0,0.07); }
[data-testid="stChatMessage"] { padding: 0.8rem 1rem; border-radius: 16px; margin-bottom: 0.4rem; }
[data-testid="stChatMessage"] p { font-size: 0.97rem; line-height: 1.6; }
[data-testid="stChatMessage"]:has(.m-user) { background: color-mix(in srgb, var(--accent) 13%, transparent); }
[data-testid="stChatMessage"]:has(.m-agent) { background: var(--tint); border: 1px solid var(--line); }
[data-testid="stChatMessage"]:has(.m-user) [data-testid="stChatMessageAvatarCustom"] { background: transparent; border: 1.5px solid var(--accent); color: var(--accent); }
[data-testid="stChatMessage"]:has(.m-agent) [data-testid="stChatMessageAvatarCustom"] { background: var(--accent); color: var(--on-accent); }
div:has(> .st-key-left_panel), div:has(> .st-key-right_panel) { height: var(--panel-h) !important; flex-basis: var(--panel-h) !important; }
.st-key-left_panel, .st-key-right_panel { height: 100% !important; }
div:has(> .st-key-chat_scroll) { height: calc(var(--panel-h) - 8.6rem) !important; flex-basis: calc(var(--panel-h) - 8.6rem) !important; }
.st-key-chat_scroll { height: 100% !important; }
</style>
"""


def group_digits(value):
    digits = re.sub(r"\D", "", value)
    return " ".join(digits[i : i + 4] for i in range(0, len(digits), 4)) or value


def describe_call(call):
    match = re.match(r"lookup_order\(order_id=(\d+)(?:, verification_value=(\w+))?\)", call)
    if match:
        if match.group(2):
            return f"Checked the caller's details for order {group_digits(match.group(1))}"
        return f"Looked up order {group_digits(match.group(1))}"
    if call.startswith("create_ticket"):
        return "Created a support ticket for a human colleague"
    return call


def field(value, name):
    if value is None:
        return None
    if isinstance(value, dict) or hasattr(value, "get"):
        try:
            return value.get(name)
        except Exception:
            pass
    return getattr(value, name, None)


@st.cache_resource(show_spinner="Loading the knowledge base and the language models...")
def load_rag():
    return Rag()


def reset():
    st.session_state.agent = Agent(load_rag())
    st.session_state.messages = []
    st.session_state.script_pos = 0
    st.session_state.turns = 0
    st.session_state.session = time.strftime("%H%M%S")
    st.session_state.pending = None
    (AUDIO_DIR / "replies").mkdir(parents=True, exist_ok=True)


def next_script_line():
    lines = SCENARIOS[st.session_state.scenario]
    if st.session_state.script_pos < len(lines):
        st.session_state.pending = lines[st.session_state.script_pos]
        st.session_state.script_pos += 1


def change_scenario():
    st.session_state.script_pos = 0


def snapshot(agent, result, heard):
    return {
        "heard": heard,
        "decision": result["decision"],
        "tools": result["tools"],
        "retrieved": result["retrieved"],
        "facts": result.get("facts", []),
    }


def add_turn(heard, result, reply_audio, voice):
    agent = st.session_state.agent
    st.session_state.messages.append({"role": "user", "text": heard or "(nothing heard)", "voice": voice})
    st.session_state.messages.append({"role": "assistant", "text": result["reply"], "audio": reply_audio, "trace": snapshot(agent, result, heard)})
    st.session_state.just_created = len(st.session_state.messages) - 1


def handle_voice(data):
    agent = st.session_state.agent
    st.session_state.turns += 1
    stem = f"{st.session_state.session}_{st.session_state.turns}"
    audio_path = AUDIO_DIR / f"{stem}.wav"
    reply_path = AUDIO_DIR / "replies" / f"{stem}.wav"
    audio_path.write_bytes(data)
    heard, result = voice_turn(audio_path, agent, reply_path)
    add_turn(heard, result, str(reply_path), True)


def handle_text(text, speak_replies):
    agent = st.session_state.agent
    st.session_state.turns += 1
    result = agent.respond(text)
    reply_audio = None
    if speak_replies:
        path = AUDIO_DIR / "replies" / f"{st.session_state.session}_{st.session_state.turns}.wav"
        try:
            reply_audio = speak(result["reply"], result.get("language", agent.language), path)
        except Exception as error:
            st.warning(f"Text-to-speech failed: {error}")
    add_turn(text, result, reply_audio, False)


def rows(items):
    cells = "".join(f'<div class="k">{html.escape(k)}</div><div class="v{" on" if on else ""}">{html.escape(str(v))}</div>' for k, v, on in items)
    return f'<div class="kv">{cells}</div>'


def show_sources(trace, rag):
    for source in trace["retrieved"]:
        if source.startswith("products.csv:"):
            product = PRODUCTS_BY_SKU.get(source.split(":")[1].upper())
            with st.expander(f"Product record {source.split(':')[1]}"):
                st.write(product or "not found")
            continue
        chunk = rag.by_id.get(source)
        if chunk:
            with st.expander(f"{chunk['title']} ({source})"):
                st.markdown(chunk["text"])


@st.dialog("Batch run results", width="medium")
def batch_dialog():
    st.caption(
        "Batch mode lets the organizers run this assistant on a whole folder of customer audio files with one command "
        "(python run.py). This table shows the saved output for each file: what was heard, the reply, the actions taken and the policy used."
    )
    results = Path("sample_outputs") / "results.csv"
    if results.exists():
        frame = pd.read_csv(results, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        st.dataframe(frame[["file", "heard", "reply", "decision"]], width="stretch", height=min(260, 42 + 36 * len(frame)))
        pick = st.selectbox("Inspect a file", list(frame.file))
        row = frame[frame.file == pick].iloc[0]
        st.markdown(f"**Actions:** {row.tools or 'none'}")
        st.markdown(f"**Policy sections used:** {row.retrieved or 'none'}")
        wav = Path("sample_outputs") / f"{Path(pick).stem}_reply.wav"
        if wav.exists():
            st.audio(str(wav))
    else:
        st.info("No batch results yet. Run this in the terminal: python run.py demo_audio --out sample_outputs --conversation")


if "agent" not in st.session_state:
    reset()
st.session_state.just_created = None
rag = load_rag()
agent = st.session_state.agent
st.markdown(STYLE, unsafe_allow_html=True)

with st.container(key="topbar"):
    brand_col, batch_col, how_col = st.columns([7.2, 1.2, 1.2], vertical_alignment="center")
    brand_col.markdown('<div class="brand"><div class="mark">Z</div><div><div class="name">ZentroMart</div><div class="sub">AI Support Assistant</div></div></div>', unsafe_allow_html=True)
    if batch_col.button("Batch results", icon=":material/folder_open:", width="stretch"):
        batch_dialog()
    with how_col.popover("How it works", icon=":material/info:", width="stretch"):
        st.markdown(
            "**From voice to reply**\n\n"
            "1. **Listen:** Gnani speech-to-text turns the customer's voice into text.\n"
            "2. **Understand:** a language model identifies the request, the language and any urgency.\n"
            "3. **Check:** the assistant searches the company policy and, once the caller is verified, the order record.\n"
            "4. **Answer:** the reply is written only from those facts. Dates and deadlines are calculated by the system.\n"
            "5. **Speak:** Gnani text-to-speech reads the reply aloud.\n\n"
            "Cases that need a person, such as high-value damage or an unrecognised order, are handed to a human colleague with a support ticket."
        )

left, center, right = st.columns([1.15, 2.9, 1.6], gap="medium")

with left.container(border=True, height=520, key="left_panel"):
    st.markdown('<div class="panel-title">Demo scenarios</div>', unsafe_allow_html=True)
    st.selectbox("Scenario", list(SCENARIOS), key="scenario", on_change=change_scenario, label_visibility="collapsed")
    lines = SCENARIOS[st.session_state.scenario]
    position = st.session_state.script_pos
    if position < len(lines):
        st.markdown(f'<div class="next"><small>Next customer message ({position + 1} of {len(lines)})</small>{html.escape(lines[position])}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="next"><small>Scenario complete</small>Start a new conversation to run it again.</div>', unsafe_allow_html=True)
    st.button("Send next message", icon=":material/send:", on_click=next_script_line, disabled=position >= len(lines), width="stretch")
    st.button("New conversation", icon=":material/refresh:", on_click=reset, width="stretch")
    speak_replies = st.toggle("Read typed replies aloud", value=True)
    st.caption("All customers, orders and policies in this demo are fictional test data. Switch between light and dark from the menu at the top right.")

with center.container(border=True, key="center_panel"):
    st.markdown('<div class="panel-title">Conversation<small>Voice and text</small></div>', unsafe_allow_html=True)
    chat_area = st.container(height=420, autoscroll=True, border=False, key="chat_scroll")
    if "accept_audio" in inspect.signature(st.chat_input).parameters:
        submission = st.chat_input("Type a message or use the microphone", accept_audio=True, audio_sample_rate=16000)
        audio_file = field(submission, "audio") if submission is not None and not isinstance(submission, str) else None
        typed = submission if isinstance(submission, str) else field(submission, "text")
    else:
        audio_file = st.audio_input("Record a voice message", sample_rate=16000)
        typed = st.chat_input("Type a message")

pending = st.session_state.pending
st.session_state.pending = None
try:
    if audio_file is not None:
        with chat_area, st.spinner("Listening..."):
            handle_voice(audio_file.getvalue())
    elif typed or pending:
        with chat_area, st.spinner("Preparing a reply..."):
            handle_text(typed or pending, speak_replies)
except Exception as error:
    st.error(f"Something went wrong: {error}")

with chat_area:
    if not st.session_state.messages:
        st.markdown('<div class="empty"><b>Start a conversation</b>Speak with the microphone in the message box, type a message, or send the next line of a demo scenario from the left.</div>', unsafe_allow_html=True)
    for index, message in enumerate(st.session_state.messages):
        avatar = ":material/person:" if message["role"] == "user" else ":material/support_agent:"
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "user" and message.get("voice"):
                st.caption("Voice message")
            marker = "m-user" if message["role"] == "user" else "m-agent"
            st.markdown(f'<span class="{marker}"></span>' + html.escape(message["text"]), unsafe_allow_html=True)
            if message.get("audio") and Path(message["audio"]).exists():
                st.audio(message["audio"], autoplay=index == st.session_state.just_created)

with right.container(border=True, height=520, key="right_panel"):
    st.markdown('<div class="panel-title">Case details</div>', unsafe_allow_html=True)
    verified = "Verified" if agent.order else ("Locked out" if agent.locked else "Not verified")
    st.markdown(
        rows(
            [
                ("Caller", verified, bool(agent.order)),
                ("Failed checks", agent.attempts, False),
                ("Order", group_digits(agent.order_id) if agent.order_id else "None yet", False),
                ("Language", LANGUAGES.get(agent.language, agent.language), False),
                ("Support ticket", "Created" if agent.ticketed else "None", agent.ticketed),
            ]
        ),
        unsafe_allow_html=True,
    )
    assistant = [m for m in st.session_state.messages if m["role"] == "assistant"]
    if assistant:
        trace = assistant[-1]["trace"]
        neutral = "" if trace["decision"] in ("answer", "escalate") else " neutral"
        st.markdown(f'<div class="label">Latest reply</div><span class="chip{neutral}">{html.escape(DECISIONS.get(trace["decision"], trace["decision"]))}</span>', unsafe_allow_html=True)
        st.markdown('<div class="label">What the system heard</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="quote">{html.escape(trace["heard"] or "(nothing heard)")}</div>', unsafe_allow_html=True)
        st.markdown('<div class="label">Actions taken</div>', unsafe_allow_html=True)
        if trace["tools"]:
            for call in trace["tools"]:
                st.markdown(f'<div class="action">{html.escape(describe_call(call))}</div>', unsafe_allow_html=True)
        else:
            st.caption("No system actions were needed for this reply.")
        if trace["facts"]:
            heading = "Reason for handover" if trace["decision"] == "escalate" else "Calculated by the system"
            st.markdown(f'<div class="label">{heading}</div>', unsafe_allow_html=True)
            for fact in trace["facts"]:
                st.markdown(f'<div class="action">{html.escape(fact)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="label">Policy used</div>', unsafe_allow_html=True)
        if trace["retrieved"]:
            show_sources(trace, rag)
        else:
            st.caption("This step was handled by the verification flow.")
        with st.expander("Technical log"):
            for message in assistant:
                for call in message["trace"]["tools"]:
                    st.code(call, language="python")
    else:
        st.caption("(Once the conversation starts, this panel shows who the caller is, what the system did and which policy it used)")