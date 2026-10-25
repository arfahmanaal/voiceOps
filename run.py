import argparse
import csv
import json
import re
import sys
from pathlib import Path

from src.agent import Agent
from src.rag import Rag
from src.speech import TROUBLE, UNCLEAR, speak, voice_turn

AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
FIELDS = ["file", "conversation", "turn", "heard", "reply", "decision", "tools", "retrieved", "language", "error"]


def natural(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def group_files(files, conversation):
    if not conversation:
        return [(path.stem, [path]) for path in files]
    groups = {}
    for path in files:
        match = re.match(r"^(.*?)[_-]?t?(\d+)$", path.stem)
        key = match.group(1) if match and match.group(1) else path.stem
        groups.setdefault(key, []).append(path)
    return [(key, sorted(paths, key=lambda p: natural(p.stem))) for key, paths in sorted(groups.items())]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the ZentroMart voice agent on a folder of customer audio files.")
    parser.add_argument("input", help="folder with customer audio files")
    parser.add_argument("--out", default="sample_outputs", help="folder for replies (text, audio, retrieved chunks)")
    parser.add_argument("--conversation", action="store_true", help="treat files named <name>_1, <name>_2 ... as turns of one conversation")
    args = parser.parse_args(argv)

    folder = Path(args.input)
    files = sorted((p for p in folder.iterdir() if p.suffix.lower() in AUDIO), key=lambda p: natural(p.name))
    if not files:
        sys.exit(f"No audio files found in {folder}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rag = Rag()
    rows = []
    for conversation_id, paths in group_files(files, args.conversation):
        agent = Agent(rag)
        for turn, path in enumerate(paths, 1):
            reply_path = out / f"{path.stem}_reply.wav"
            row = {"file": path.name, "conversation": conversation_id, "turn": turn, "heard": "", "reply": "", "decision": "", "tools": "", "retrieved": "", "language": agent.language, "error": ""}
            try:
                heard, result = voice_turn(path, agent, reply_path)
                row.update(
                    heard=heard,
                    reply=result["reply"],
                    decision=result["decision"],
                    tools=" ; ".join(result["tools"]),
                    retrieved=" ; ".join(result["retrieved"]),
                    language=result.get("language", agent.language),
                )
            except Exception as error:
                language = agent.language
                row.update(error=str(error)[:300], reply=TROUBLE.get(language, TROUBLE["en"]), decision="error")
                try:
                    speak(row["reply"], language, reply_path)
                except Exception as speech_error:
                    row["error"] += f" | tts: {str(speech_error)[:200]}"
            (out / f"{path.stem}.txt").write_text(row["reply"], encoding="utf-8")
            (out / f"{path.stem}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)
            print(f"{path.name}: {row['decision'] or 'ok'} | heard: {row['heard'][:60]!r}", flush=True)

    with open(out / "results.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"done: {len(rows)} files, results in {out}")


if __name__ == "__main__":
    main()