# ZentroMart VoiceOps Agent

A customer-support agent for the fictional store ZentroMart (VoiceOps 2026).
It answers from the company documents with retrieval (RAG), holds a multi-turn conversation, verifies the caller before sharing order details, uses the `lookup_order` and `create_ticket` tools, and talks through Gnani speech-to-text and text-to-speech.

Repo: https://github.com/arfahmanaal/voiceOps/

## How it works

```
customer audio -> Gnani STT -> transcript cleanup -> agent -> Gnani TTS -> spoken reply
                                                      |
                              LLM + retrieval over kb/*.md + order tools + date rules
```

- **Step 1, prompt-only baseline** (`src/baseline.py`): one LLM call with a ZentroMart system prompt, no documents.
- **Step 2, RAG** (`src/kb.py`, `src/retriever.py`, `src/rag.py`): the 21 knowledge-base documents are split into 84 sections, each keeping its section ID, status and effective date. Sections are embedded with `intfloat/multilingual-e5-small` and searched by cosine similarity. Superseded sections (returns v3) are used only when the question concerns a date before 1 August 2026. Business-day and refund-date arithmetic is done in code (`src/dates.py`), never by the LLM.
- **Step 3, conversation** (`src/agent.py`): code handles order-ID parsing, caller verification (last 4 phone digits or pincode, one retry, then lockout), ticket creation, and escalation rules (human requested, legal threat, unauthorized activity, safety risk, damage claim above INR 10,000, third contact in 7 days, refund more than 3 business days overdue). The LLM only writes the wording and is told what code actually did.
- **Step 4, voice** (`src/speech.py`, `src/textnorm.py`): Gnani STT (English-India, accepts Hindi mixed in) and TTS. Spoken numbers, SKUs and "u p i" are normalized before the agent sees the text. Replies are made speech-friendly (long numbers grouped, rupee amounts spelled out). TTS audio is cached.

## Setup

Requires Python 3.10 or higher and an internet connection. The first run downloads the embedding model (about 470 MB).

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the keys. Never commit `.env`.

| Variable | Meaning |
|---|---|
| `GNANI_API_KEY` | Gnani key (covers STT and TTS), from app.gnani.ai/voice |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | Any OpenAI-compatible chat endpoint (tested setup: Groq, `openai/gpt-oss-120b`) |
| `LLM_FALLBACK_MODEL` | Optional comma-separated backup model(s) used when the main one is rate limited |
| `LLM_REASONING_EFFORT` | Optional, for reasoning models (`low` recommended) |
| `DATASET_DIR` | Folder with the provided dataset (`kb/`, `data/`, `tools/`, `practice/`); default `dataset` |
| `GNANI_STT_LANGUAGE`, `GNANI_VOICE_EN`, `GNANI_VOICE_HI` | Optional speech settings (defaults: `en-IN`, `Kaveri`, `Nalini`) |

The dataset is **not** included in this repository. Extract the organizers' dataset so that `DATASET_DIR` points to the folder containing `kb/`, `data/` and `tools/` (the code imports `lookup_order` and `create_ticket` from that `tools` package). The mock ticket tool writes `tickets_created.csv` inside it.

## Run

Batch run on a folder of customer audio (the organizers' command):

```
python run.py <audio_folder> --out sample_outputs
```

- Supported audio: `.wav .mp3 .flac .ogg .m4a .aac`, up to 60 seconds per file.
- By default every file is a separate customer. If files are numbered turns of one call (`call1_1.wav`, `call1_2.wav`, ...), add `--conversation` to group them into one session.
- For each file it saves `<name>.txt` (reply text), `<name>_reply.wav` (reply audio) and `<name>.json` (transcript, reply, decision, tool calls and the retrieved document chunks). `results.csv` summarizes all files.
- Unclear or empty audio gets a polite "please say that again" reply. Errors never stop the batch; the file gets a fallback spoken reply and the error is recorded.

Talk to the agent with a microphone (Windows):

```
python demo.py
```

Other scripts: `voice_test.py` (hear the three voices), `check_setup.py` (keys and tools), `evaluate_retrieval.py`, `evaluate.py baseline|rag|agent question`, `evaluate_conv.py` (practice evaluations).

## Results on the practice data

Measured with my own approximation of the scoring (key facts included, forbidden phrases absent, expected tool calls), so they are indicative, not official.

| Measure | Result |
|---|---|
| Step 1, prompt only, 52 questions | 17/52 |
| Step 2, RAG, 52 questions | 47/52 |
| Retrieval (hit / recall of the expected sections) | 0.98 / 0.91 |
| Conversations, 96 turns | 84/96 measured; tool calls matched on all 96 turns |

The conversation figure includes 11 first turns that were saved before a fix (the AI-assistant introduction). The fix was verified on the 12 conversations that were rerun; the full rerun was skipped to save free-tier tokens. <update with your final numbers>

## Design decisions and assumptions

- Documents are authoritative in this order: current policy, agent guide, help article, FAQ, marketing. Support logs are never used as policy.
- "Today" is 20 October 2026, as the brief specifies. Business days are Monday to Saturday excluding `holidays.csv`; return windows use calendar days.
- Order details are shown to the model only after verification, and without phone, pincode or address.
- The agent never says an action happened unless its code did it (a ticket is confirmed only when `created=True`).
- Assumption: in the organizers' batch run each audio file is one customer turn.

## Known limitations

- Free-tier LLM limits make full evaluation runs slow; the client waits and falls back to a smaller model automatically.
- Hinglish replies are spoken with a Hindi voice, which may mispronounce some English words. <update after your voice test>
- The refund timeline for refused parcels is occasionally left out of the reply in English conversations.
- Retrieval and prompts were tuned on the 52 practice questions, so scores on the hidden test may be lower.

## Project layout

```
run.py            batch run on a folder of audio
demo.py           microphone demo
src/              config, llm, kb, dates, retriever, rag, baseline, agent, speech, textnorm, scoring
evaluate*.py      practice evaluations
sample_outputs/   example outputs from run.py
```