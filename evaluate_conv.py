import json
import os
import re
import sys

import pandas as pd

from src.agent import Agent
from src.config import DATASET_DIR
from src.rag import Rag
from src.scoring import check

PARTIAL = os.path.join("outputs", "conversations_partial.json")


def tools_match(expected, actual):
    wanted = [re.sub(r"\s+", "", call) for call in expected.split(";") if call.strip()]
    got = [re.sub(r"\s+", "", call) for call in actual]
    wanted_names = sorted(call.split("(")[0] for call in wanted if call.startswith("create_ticket"))
    got_names = sorted(call.split("(")[0] for call in got if call.startswith("create_ticket"))
    lookups_wanted = [call for call in wanted if call.startswith("lookup_order")]
    lookups_got = [call for call in got if call.startswith("lookup_order")]
    return wanted_names == got_names and lookups_wanted == lookups_got


os.makedirs("outputs", exist_ok=True)
if "--fresh" in sys.argv and os.path.exists(PARTIAL):
    os.remove(PARTIAL)
only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
rows = []
if os.path.exists(PARTIAL):
    with open(PARTIAL, encoding="utf-8") as handle:
        rows = json.load(handle)
if only:
    rows = [row for row in rows if row["conversation_id"] not in only]
finished = {row["conversation_id"] for row in rows}
if finished:
    print(f"resuming: {len(finished)} conversations already saved", flush=True)

conversations = pd.read_csv(os.path.join(DATASET_DIR, "practice", "conversations.csv"), dtype=str).fillna("")
conversations["turn"] = conversations.turn.astype(int)
rag = Rag()

for conversation_id, group in conversations.groupby("conversation_id"):
    if conversation_id in finished or (only and conversation_id not in only):
        continue
    agent = Agent(rag)
    new_rows = []
    for _, row in group.sort_values("turn").iterrows():
        result = agent.respond(row.customer_message)
        include_ok, exclude_ok = check(result["reply"], row.must_include, row.must_not_include)
        tools_ok = tools_match(row.expected_tool_calls, result["tools"])
        new_rows.append(
            {
                "conversation_id": conversation_id,
                "turn": int(row.turn),
                "scenario": row.scenario,
                "customer": row.customer_message,
                "reply": result["reply"],
                "decision": result["decision"],
                "expected_decision": row.expected_decision,
                "tools": " ; ".join(result["tools"]),
                "expected_tools": row.expected_tool_calls,
                "retrieved": " ; ".join(result["retrieved"]),
                "include_ok": bool(include_ok),
                "exclude_ok": bool(exclude_ok),
                "tools_ok": bool(tools_ok),
                "decision_ok": result["decision"] == row.expected_decision,
                "passed": bool(include_ok and exclude_ok and tools_ok),
            }
        )
    rows.extend(new_rows)
    with open(PARTIAL, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False)
    print(conversation_id, "done", flush=True)

frame = pd.DataFrame(rows).sort_values(["conversation_id", "turn"])
frame.to_csv(os.path.join("outputs", "conversations.csv"), index=False, encoding="utf-8-sig")

per_conversation = frame.groupby("conversation_id").passed.all()
print(f"\nturns passed: {frame.passed.sum()}/{len(frame)} ({frame.passed.mean():.2f})")
print(f"conversations fully passed: {per_conversation.sum()}/{len(per_conversation)}")
print(f"decision matches (informational): {frame.decision_ok.sum()}/{len(frame)}")
print(frame.groupby("scenario").passed.agg(["sum", "count"]).to_string())
print("\nfailures:")
for _, row in frame[~frame.passed].iterrows():
    print(f"{row.conversation_id} t{row.turn} {row.scenario} | include {row.include_ok} exclude {row.exclude_ok} tools {row.tools_ok} | decision {row.decision} vs {row.expected_decision}")
    print("  C:", row.customer[:90])
    print("  A:", row.reply[:260].replace("\n", " "))
    if not row.tools_ok:
        print("  tools:", row.tools, "| expected:", row.expected_tools)