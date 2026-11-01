import os
import re

import pandas as pd

from src.config import DATASET_DIR
from src.kb import find_products
from src.rag import refund_fact
from src.speech import clean_transcript

questions = pd.read_csv(os.path.join(DATASET_DIR, "practice", "questions.csv"), dtype=str).fillna("")

for column in ["question", "asr_text"]:
    products_total = products_found = dates_total = dates_ok = 0
    for _, row in questions.iterrows():
        text = clean_transcript(row[column]) if column == "asr_text" else row[column]
        for gold in [item.strip() for item in row.gold_evidence.split(";") if item.strip().startswith("products.csv:")]:
            products_total += 1
            products_found += gold in ["products.csv:" + product["sku"] for product in find_products(text)]
        if row.intent == "refund_timeline" and re.search(r"\d{4}-\d{2}-\d{2}", row.must_include):
            expected = pd.Timestamp(re.search(r"\d{4}-\d{2}-\d{2}", row.must_include).group())
            fact = refund_fact(text)
            dates_total += 1
            dates_ok += bool(fact) and f"{expected.day} {expected:%B %Y}" in fact
    print(f"{column:9s} product rows found: {products_found}/{products_total} | refund dates computed correctly: {dates_ok}/{dates_total}")