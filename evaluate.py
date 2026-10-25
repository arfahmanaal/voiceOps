import json
import os
import sys

import pandas as pd

from src.config import DATASET_DIR
from src.scoring import check

MODE = sys.argv[1] if len(sys.argv) > 1 else "rag"
COLUMN = sys.argv[2] if len(sys.argv) > 2 else "question"
PARTIAL = os.path.join("outputs", f"{MODE}_{COLUMN}_partial.json")

if MODE == "baseline":
    from src.baseline import answer
elif MODE == "agent":
    from src.agent import Agent
    from src.rag import Rag

    shared = Rag()

    def answer(question):
        return Agent(shared).respond(question)
else:
    from src.rag import Rag

    answer = Rag().answer

os.makedirs("outputs", exist_ok=True)
if "--fresh" in sys.argv and os.path.exists(PARTIAL):
    os.remove(PARTIAL)
rows = []
if os.path.exists(PARTIAL):
    with open(PARTIAL, encoding="utf-8") as handle:
        rows = json.load(handle)
finished = {row["id"] for row in rows}
if finished:
    print(f"resuming: {len(finished)} questions already saved", flush=True)

questions = pd.read_csv(os.path.join(DATASET_DIR, "practice", "questions.csv"), dtype=str).fillna("")

for number, (_, row) in enumerate(questions.iterrows(), 1):
    if row.id in finished:
        continue
    result = answer(row[COLUMN])
    include_ok, exclude_ok = check(result["reply"], row.must_include, row.must_not_include)
    rows.append(
        {
            "id": row.id,
            "intent": row.intent,
            "language": row.language,
            "question": row[COLUMN],
            "reply": result["reply"],
            "retrieved": " ; ".join(result["retrieved"]),
            "include_ok": bool(include_ok),
            "exclude_ok": bool(exclude_ok),
            "passed": bool(include_ok and exclude_ok),
        }
    )
    with open(PARTIAL, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False)
    print(f"{number}/{len(questions)} {row.id} {'PASS' if rows[-1]['passed'] else 'FAIL'}", flush=True)

result_frame = pd.DataFrame(rows)
result_frame.to_csv(os.path.join("outputs", f"{MODE}_{COLUMN}.csv"), index=False, encoding="utf-8-sig")

print(f"\n{MODE} on '{COLUMN}': {result_frame.passed.sum()}/{len(result_frame)} passed ({result_frame.passed.mean():.2f})")
print(result_frame.groupby("intent").passed.agg(["sum", "count"]).to_string())
print("\nfailures:")
for _, row in result_frame[~result_frame.passed].iterrows():
    print(row.id, row.intent, "| include_ok:", row.include_ok, "| exclude_ok:", row.exclude_ok)
    print("  Q:", row.question[:100])
    print("  A:", row.reply[:250].replace("\n", " "))