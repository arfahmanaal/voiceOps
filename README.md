# ZentroMart VoiceOps Agent

A customer-support agent for the fictional store ZentroMart (VoiceOps 2026).
It answers from the company documents with retrieval (RAG), holds a multi-turn conversation, verifies the caller before sharing order details, uses the `lookup_order` and `create_ticket` tools, and talks through Gnani speech-to-text and text-to-speech.

Repo: https://github.com/arfahmanaal/voiceOps/

## What it does
 
| Capability | How |
|---|---|
| Answers from the documents | 21 knowledge-base files split into 84 sections (section ID, status and effective date kept), embedded with `multilingual-e5-small`, searched by cosine similarity. Answers use only retrieved sections and say "I don't know" when the documents are silent. |
| Handles Hindi, Hinglish and English | An LLM rewrites each message into an English search query; replies follow the customer's language and script. |
| Verifies the caller | Order ID, then last 4 phone digits or delivery pincode. One retry, then lockout. Order details are never shown before verification. |
| Uses the tools | `lookup_order` and `create_ticket`. A handover is confirmed only when the ticket tool returns `created=True`. |
| Hands over to a human | Human requested, legal threat, unauthorized activity, safety risk, damage claim above INR 10,000, third contact within 7 days, refund more than 3 business days late (reference number shared). |
| Computes dates in code | Business days (Monday to Saturday, no warehouse holidays), refund windows and return deadlines never come from the LLM. "Today" is 20 October 2026, as the brief specifies. |
| Speaks | Gnani STT in, Gnani TTS out, with spoken-number cleanup and speech-friendly replies. |
 
**Design principle: code decides, the LLM only writes the wording.**
 
## Results on the practice data
 
Measured with my own scorer (key facts present, forbidden phrases absent, expected tool calls made), so these numbers are indicative, not official.
 
| Measure | Result |
|---|---|
| Step 1: prompt only, 52 questions | 51/52 |
| Step 2: RAG, same questions | 47/52 |
| Retrieval hit / recall / MRR | 0.98 / 0.91 / 0.75 (first version: 0.72 / 0.57 / 0.52) |
| Practice conversations (96 turns) | 84/96; expected tool calls matched on all 96 turns |
| Speech-style practice text | 13/13 product rows found, 9/9 refund dates computed |
 

## Setup
 
Requires **Python 3.10 or higher** and an internet connection. The first run downloads the embedding model (about 470 MB).
 
```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```
 
Copy `.env.example` to `.env` and fill in the keys. **Never commit `.env`.**
 
| Variable | Meaning |
|---|---|
| `GNANI_API_KEY` | Gnani key for STT and TTS (app.gnani.ai/voice) |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | Any OpenAI-compatible chat endpoint (developed with Groq and `openai/gpt-oss-120b`) |
| `LLM_FALLBACK_MODEL` | Optional comma-separated backup models used when the main one is rate limited |
| `LLM_REASONING_EFFORT` | Optional, for reasoning models (`low` recommended) |
| `LLM_TPM` | Optional tokens-per-minute budget for pacing (default 7000) |
| `RETRIEVE_K` | Optional number of retrieved sections (default 5) |
| `DATASET_DIR` | Folder containing the provided dataset (`kb/`, `data/`, `tools/`, `practice/`); default `dataset` |
| `GNANI_STT_LANGUAGE`, `GNANI_VOICE_EN`, `GNANI_VOICE_HI` | Optional speech settings (defaults `en-IN`, `Kaveri`, `Nalini`) |
 
**The dataset is not part of this repository.** Extract the organizers' dataset so that `DATASET_DIR` points at the folder that contains `kb/`, `data/` and `tools/`. The code imports `lookup_order` and `create_ticket` from that `tools` package, and the mock ticket tool writes `tickets_created.csv` inside it.
 
Check your setup:
 
```
python check_setup.py
```
 
## Run
 
### Batch run on a folder of customer audio (the organizers' command)
 
```
python run.py <audio_folder> --out sample_outputs
```
 
- Supported audio: `.wav .mp3 .flac .ogg .m4a .aac`, up to 60 seconds per file.
- Every file is handled as a separate customer. If files are numbered turns of one call (`call1_1.wav`, `call1_2.wav`, ...), add `--conversation` to put them in one session.
- For each file it saves `<name>.txt` (reply text), `<name>_reply.wav` (reply audio) and `<name>.json` (transcript, reply, decision, tool calls and the retrieved document sections). `results.csv` summarizes the whole run.
- Unclear or empty audio gets a polite "please say that again". An error never stops the batch: the file receives a spoken fallback reply and the error is recorded.
### Demo app (microphone in the browser)
 
```
streamlit run app.py --server.fileWatcherType none
```
 
Opens at http://localhost:8501. Use the microphone in the message box, or type. Demo scenarios on the left send ready-made customer messages. Switch between light and dark from the three-dots menu at the top right. Batch results opens the saved output of `run.py`. The "Case details" panel shows the caller status, what was heard, the actions taken, the dates calculated by the system and the policy sections used. Recordings are saved to `demo_audio/`, which can be fed straight to `run.py`.
 
### Other scripts
 
| Script | Purpose |
|---|---|
| `demo.py` | Command-line microphone demo |
| `voice_test.py` | Synthesizes sample replies so you can listen to the voices |
| `evaluate.py baseline question` / `rag question` / `agent question` | Step 1 vs Step 2 evaluation on the practice questions |
| `evaluate_conv.py` | Practice conversations (`--fresh` restarts, `--only PC002,PC009` reruns some) |
| `evaluate_retrieval.py` | Retrieval hit, recall and MRR against the expected sections |
| `evaluate_speech_text.py` | Checks the spoken-text cleanup on the speech-style practice text (no LLM calls) |
 
Evaluations are resumable: if the LLM provider rate limits you, rerun the same command and it continues.
 
## How it works
 
```
customer audio -> Gnani STT -> transcript cleanup -> agent -> Gnani TTS -> spoken reply
                                                       |
                  one LLM call analyses the message (language, intent flags, search query)
                  retrieval over the knowledge base + order tools + date rules in code
                  one LLM call writes the reply from facts that code supplies
```
 
| Module | Role |
|---|---|
| `src/kb.py`, `src/retriever.py` | Section-aware knowledge base and embedding search |
| `src/rag.py`, `src/baseline.py` | Step 2 and Step 1 answerers |
| `src/agent.py` | Conversation state, verification, tools, escalation rules |
| `src/dates.py` | Business days, refund windows, return deadlines |
| `src/textnorm.py`, `src/speech.py` | Spoken-number cleanup, Gnani STT and TTS, reply cache |
| `src/llm.py` | OpenAI-compatible client with pacing, retry and fallback models |
| `src/scoring.py` | Local scorer used by the evaluation scripts |
 
Key decisions:
 
- Document precedence: current policy, agent guide, help article, FAQ, marketing. Support logs are never treated as policy.
- The superseded returns policy (v3) is used only when a question concerns a date before 1 August 2026.
- The model sees order facts without phone number, pincode or address.
- Only synthetic data is used. No real customer data or recordings.
## Known limitations
 
- Scores use my own scorer on practice data that I also tuned on, so hidden-test scores may be lower.
- Free-tier LLM limits make full evaluations slow; the client paces itself and can fall back to a smaller model.
- Hinglish replies are spoken with a Hindi voice, which may mispronounce some English words.
- In batch mode, each audio file is one customer turn unless `--conversation` is used.
## Project structure
 
```
README.md  .env.example  requirements.txt  run.py  app.py  demo.py
src/                  agent, rag, retriever, kb, dates, speech, textnorm, llm, scoring, baseline, config
evaluate*.py  check_setup.py  voice_test.py
.streamlit/           theme for the demo app
docs/screenshots/     images used in this README
sample_outputs/       example output of run.py
report.pdf  demo.mp4
```
 