import hashlib
import tempfile
import time
from pathlib import Path

import streamlit as st

from src.agent import Agent
from src.kb import PRODUCTS_BY_SKU
from src.rag import Rag
from src.speech import speak, voice_turn

st.set_page_config(page_title="ZentroMart AI Support", page_icon="🎧", layout="wide")

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

with st.sidebar:
    st.header("Demo controls")
    speak_replies = st.toggle("Speak replies for typed messages", value=True)
    st.button("New customer", on_click=reset, use_container_width=True)
    st.divider()
    st.selectbox("Demo script", list(SCRIPTS), key="script_name", on_change=change_script)
    lines = SCRIPTS[st.session_state.script_name]
    position = st.session_state.script_pos
    if position < len(lines):
        st.caption(f"Next line ({position + 1}/{len(lines)}): {lines[position]}")
    else:
        st.caption("Script finished. Press New customer to restart.")
    st.button("Send next line", on_click=next_script_line, use_container_width=True, disabled=position >= len(lines))
    st.divider()
    st.caption("Pipeline: customer audio, Gnani STT, agent (LLM + retrieval + tools), Gnani TTS, spoken reply.")

st.title("ZentroMart AI Support")
left, right = st.columns([3, 2], gap="large")

with left:
    audio = st.audio_input("Speak to the agent", key=f"mic{st.session_state.mic_key}", sample_rate=16000)
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

with left:
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🎧"):
            if message["role"] == "user" and message.get("voice"):
                st.caption("heard from your voice")
            st.write(message["text"])
            if message.get("audio") and Path(message["audio"]).exists():
                st.audio(message["audio"], autoplay=index == st.session_state.just_created)

with right:
    st.subheader("Behind the scenes")
    status = st.columns(3)
    status[0].metric("Customer verified", "yes" if agent.order else ("locked out" if agent.locked else "no"))
    status[1].metric("Failed checks", agent.attempts)
    status[2].metric("Ticket created", "yes" if agent.ticketed else "no")
    st.caption(f"Order ID in session: {agent.order_id or 'none yet'} | reply language: {agent.language}")
    assistant = [m for m in st.session_state.messages if m["role"] == "assistant"]
    if assistant:
        trace = assistant[-1]["trace"]
        st.markdown(f"**Latest decision:** `{trace['decision']}`")
        st.markdown("**What the customer said (after speech-to-text and cleanup):**")
        st.code(trace["heard"] or "(nothing heard)", language=None)
        st.markdown("**Tool calls this turn:**")
        if trace["tools"]:
            for call in trace["tools"]:
                st.code(call, language="python")
        else:
            st.caption("none")
        if trace["facts"]:
            st.markdown("**Computed in code (not by the LLM):**")
            for fact in trace["facts"]:
                st.info(fact)
        st.markdown("**Policy sections the answer was based on:**")
        if trace["retrieved"]:
            show_sources(trace, rag)
        else:
            st.caption("none for this step (handled by the verification flow)")
        with st.expander("All tool calls in this conversation"):
            for message in assistant:
                for call in message["trace"]["tools"]:
                    st.code(call, language="python")
    else:
        st.info("Speak, type, or use the demo script in the sidebar. Every step the agent takes shows up here.")