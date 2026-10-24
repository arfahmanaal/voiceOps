import os
import sys

import pandas as pd

from src.config import DATASET_DIR
from src.rag import Rag

COLUMN = sys.argv[1] if len(sys.argv) > 1 else "question"

questions = pd.read_csv(os.path.join(DATASET_DIR, "practice", "questions.csv"), dtype=str).fillna("")
rag = Rag()

scores = {"searched": [0, 0.0, 0.0], "used": [0, 0.0, 0.0]}
total = 0
misses = []

for _, row in questions.iterrows():
    gold = [item.strip() for item in row.gold_evidence.split(";") if item.strip()]
    if not gold:
        continue
    found = rag.retrieve(row[COLUMN])
    total += 1
    for name in scores:
        ids = found[name]
        got = [item for item in gold if item in ids]
        rank = next((i + 1 for i, item in enumerate(ids) if item in gold), 0)
        scores[name][0] += bool(got)
        scores[name][1] += len(got) / len(gold)
        scores[name][2] += 1 / rank if rank else 0
    missing = [item for item in gold if item not in found["used"]]
    if missing:
        misses.append((row.id, row[COLUMN][:70], missing, found["used"]))

print(f"questions with evidence: {total}")
for name, (hits, recall, rr) in scores.items():
    print(f"{name:9s} hit {hits / total:.2f}  recall {recall / total:.2f}  MRR {rr / total:.2f}")
print(f"\nstill missing gold after everything: {len(misses)}")
for qid, text, missing, ids in misses:
    print(qid, text, "| missing:", missing, "| got:", ids)