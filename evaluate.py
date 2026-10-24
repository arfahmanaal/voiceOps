import os
import sys

import pandas as pd

from src.config import DATASET_DIR
from src.scoring import check

MODE = sys.argv[1] if len(sys.argv) > 1 else "rag"
COLUMN = sys.argv[2] if len(sys.argv) > 2 else "question"

if MODE == "baseline":
    from src.baseline import answer
else:
    from src.rag import Rag

    answer = Rag().answer

questions = pd.read_csv(os.path.join(DATASET_DIR, "practice", "questions.csv"), dtype=str).fillna("")
rows = []

for number, (_, row) in enumerate(questions.iterrows(), 1):
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
            "include_ok": include_ok,
            "exclude_ok": exclude_ok,
            "passed": include_ok and exclude_ok,
        }
    )
    print(f"{number}/{len(questions)} {row.id} {'PASS' if rows[-1]['passed'] else 'FAIL'}", flush=True)

result_frame = pd.DataFrame(rows)
os.makedirs("outputs", exist_ok=True)
result_frame.to_csv(os.path.join("outputs", f"{MODE}_{COLUMN}.csv"), index=False, encoding="utf-8-sig")

print(f"\n{MODE} on '{COLUMN}': {result_frame.passed.sum()}/{len(result_frame)} passed ({result_frame.passed.mean():.2f})")
print(result_frame.groupby("intent").passed.agg(["sum", "count"]).to_string())
print("\nfailures:")
for _, row in result_frame[~result_frame.passed].iterrows():
    print(row.id, row.intent, "| include_ok:", row.include_ok, "| exclude_ok:", row.exclude_ok)
    print("  Q:", row.question[:100])
    print("  A:", row.reply[:250].replace("\n", " "))