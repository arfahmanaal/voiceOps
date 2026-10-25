import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from src.agent import Agent
from src.rag import Rag
from src.speech import voice_turn

try:
    import winsound
except ImportError:
    winsound = None

RATE = 16000
MAX_SECONDS = 55
FOLDER = Path(os.getenv("DEMO_DIR", "demo_audio"))


def record(path):
    frames = []

    def callback(indata, count, info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", callback=callback):
        input("  recording... press Enter to stop ")
    audio = np.concatenate(frames) if frames else np.zeros((0, 1), dtype="int16")
    audio = audio[: MAX_SECONDS * RATE]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(audio.tobytes())


def play(path):
    if winsound:
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    else:
        print(f"  reply audio saved at {path}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    FOLDER.mkdir(exist_ok=True)
    (FOLDER / "replies").mkdir(exist_ok=True)
    agent = Agent(Rag())
    session = time.strftime("%H%M%S")
    turn = 0
    print("ZentroMart voice agent. Press Enter, speak, press Enter again. Type q then Enter to quit.")
    while True:
        if input("\nReady> ").strip().lower() == "q":
            break
        turn += 1
        audio_path = FOLDER / f"{session}_{turn}.wav"
        reply_path = FOLDER / "replies" / f"{session}_{turn}.wav"
        record(audio_path)
        heard, result = voice_turn(audio_path, agent, reply_path)
        print("  you said:", heard or "(nothing heard)")
        print("  agent   :", result["reply"])
        if result.get("tools"):
            print("  tools   :", ", ".join(result["tools"]))
        play(reply_path)


if __name__ == "__main__":
    main()