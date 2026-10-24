import os
import sys

import pandas as pd

from src.config import DATASET_DIR
from src.kb import find_products
from src.retriever import Retriever

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
COLUMN = sys.argv[2] if len(sys.argv) > 2 else "question"

questions = pd.read_csv(os.path.join(DATASET_DIR, "practice", "questions.csv"), dtype=str).fillna("")
retriever = Retriever()

total = hits = 0
recall_sum = rr_sum = 0.0
misses = []

for _, row in questions.iterrows():
    gold = [item.strip() for item in row.gold_evidence.split(";") if item.strip()]
    if not gold:
        continue
    text = row[COLUMN]
    ids = [chunk["id"] for chunk, _ in retriever.search(text, K)]
    ids += ["products.csv:" + product["sku"] for product in find_products(text)]
    found = [item for item in gold if item in ids]
    rank = next((i + 1 for i, item in enumerate(ids) if item in gold), 0)
    total += 1
    hits += bool(found)
    recall_sum += len(found) / len(gold)
    rr_sum += 1 / rank if rank else 0
    if len(found) < len(gold):
        misses.append((row.id, row[COLUMN][:70], gold, ids))

print(f"questions with evidence: {total}")
print(f"hit@{K} (any gold found): {hits / total:.2f}")
print(f"recall@{K} (all gold found): {recall_sum / total:.2f}")
print(f"MRR: {rr_sum / total:.2f}")
print(f"\nmisses: {len(misses)}")
for qid, text, gold, ids in misses:
    print(qid, text, "| gold:", gold, "| got:", ids)