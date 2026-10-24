import json
import os
import re
from datetime import date

from src.dates import REFUND_DAYS, refund_window
from src.kb import find_products
from src.llm import chat
from src.retriever import Retriever

K = int(os.getenv("RETRIEVE_K", "5"))
ALWAYS = ["KB-00.1", "KB-00.3", "ESC-1"]
CACHE_FILE = os.path.join(".cache", "rewrites.json")
SUPERSEDED_BEFORE = date(2026, 8, 1)

DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "जनवरी": 1, "फरवरी": 2, "फ़रवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6,
    "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "अक्तूबर": 10,
    "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
}
DATE = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(sorted(map(re.escape, MONTHS), key=len, reverse=True)) + r")(?:\s+(\d{4}))?",
    re.I,
)
METHODS = [
    ("net banking", "netbanking"), ("netbanking", "netbanking"), ("नेट बैंकिंग", "netbanking"),
    ("upi", "upi"), ("यूपीआई", "upi"),
    ("credit card", "card"), ("debit card", "card"), ("क्रेडिट कार्ड", "card"), ("डेबिट कार्ड", "card"), ("card", "card"), ("कार्ड", "card"),
    ("wallet", "wallet"), ("वॉलेट", "wallet"),
]
LABELS = {"upi": "UPI", "card": "credit or debit card", "netbanking": "net banking", "wallet": "ZentroMart Wallet"}

SYSTEM = (
    "You are ZentroMart's AI support assistant. Answer the customer using ONLY the CONTEXT, COMPUTED FACTS and RULES given.\n"
    "- Say once that you are ZentroMart's AI support assistant.\n"
    "- If the context does not cover the question, say you do not have that information and offer to connect the customer to a human colleague. Never invent a policy, date, amount or fee.\n"
    "- Ask one short clarifying question only when the answer truly depends on a missing detail. Otherwise answer from the context and state the condition, for example: for items delivered on or after 1 August 2026 it is 7 days.\n"
    "- When documents disagree, follow KB-00.1: current policy beats agent guide, help article, FAQ and marketing. A document with status superseded applies only where it says so.\n"
    "- Use COMPUTED FACTS exactly as given. Do not do date arithmetic yourself.\n"
    "- Reply in the customer's language and script: English, Hindi in Devanagari, or Hinglish in Roman letters.\n"
    "- Reply in two or three short sentences. Include the specific numbers from the policy (days, months, amounts, fees) and write numbers as digits, for example 3, not three. Write dates with the month name and the year.\n"
    "- Never promise compensation, refunds or exceptions that a policy does not allow. Never say an action has been done.\n"
    "- If ESC-1 applies, say a human colleague will contact the customer."
)
REWRITE_SYSTEM = (
    "Rewrite the customer message as one short English search query (at most 15 words) for a retail support knowledge base. "
    "Keep product names and payment methods. Remove dates, order numbers and greetings. Output only the query."
)


def fmt(day):
    return f"{day.day} {day:%B %Y}"


def refund_fact(text):
    text = text.translate(DIGITS)
    lowered = text.lower()
    match = DATE.search(text)
    method = next((value for key, value in METHODS if key in lowered), None)
    if not match or not method:
        return None
    start = date(int(match.group(3) or 2026), MONTHS[match.group(2).lower()], int(match.group(1)))
    low, high = REFUND_DAYS[method]
    if high == 0:
        return f"Refund to {LABELS[method]} initiated on {fmt(start)} arrives instantly, within 2 hours."
    earliest, latest = refund_window(start.isoformat(), method)
    return (
        f"Refund by {LABELS[method]} initiated on {fmt(start)} takes {low} to {high} business days "
        f"(Sundays and warehouse holidays not counted): between {fmt(earliest)} and {fmt(latest)}. "
        f"Tell the customer it should arrive by {fmt(latest)}."
    )


def describe_chunk(chunk):
    header = f"[{chunk['id']}] {chunk['doc_title']} ({chunk['doc_type']}, status={chunk['status']}, effective_from={chunk['effective_from']})"
    return header + "\n" + chunk["text"]


def describe_product(product):
    return (
        f"[products.csv:{product['sku']}] title={product['title']}; return_category={product['return_category']}; "
        f"price_inr={product['price_inr']}; fulfilled_by={product['fulfilled_by']}; "
        f"assured={product['assured']}; warranty_months={product['warranty_months']}"
    )


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=1)


def strip_dates(text):
    text = DATE.sub(" ", text.translate(DIGITS))
    text = re.sub(r"\b20\d{2}\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def mentions_old_date(text):
    for match in DATE.finditer(text.translate(DIGITS)):
        day = date(int(match.group(3) or 2026), MONTHS[match.group(2).lower()], int(match.group(1)))
        if day < SUPERSEDED_BEFORE:
            return True
    return False


class Rag:
    def __init__(self, retriever=None):
        self.retriever = retriever or Retriever()
        self.by_id = {chunk["id"]: chunk for chunk in self.retriever.chunks}
        self.cache = load_cache()

    def rewrite(self, question):
        if question not in self.cache:
            self.cache[question] = chat(
                REWRITE_SYSTEM, [{"role": "user", "content": question}], temperature=0, max_tokens=400
            ).strip()
            save_cache(self.cache)
        return self.cache[question]

    def retrieve(self, question):
        queries = [self.rewrite(question), strip_dates(question) or question]
        hits = [
            chunk
            for chunk, _ in self.retriever.search(queries, K, include_superseded=mentions_old_date(question))
        ]
        hit_ids = [chunk["id"] for chunk in hits]
        rules = [self.by_id[i] for i in ALWAYS if i in self.by_id and i not in hit_ids]
        products = find_products(question)
        return {
            "hits": hits,
            "rules": rules,
            "products": products,
            "searched": hit_ids + ["products.csv:" + product["sku"] for product in products],
            "used": hit_ids + [chunk["id"] for chunk in rules] + ["products.csv:" + product["sku"] for product in products],
        }

    def answer(self, question):
        found = self.retrieve(question)
        facts = [fact for fact in [refund_fact(question)] if fact]
        context = "\n\n".join(
            [describe_chunk(chunk) for chunk in found["hits"]]
            + [describe_product(product) for product in found["products"]]
        )
        rule_text = "\n\n".join(describe_chunk(chunk) for chunk in found["rules"])
        message = (
            f"CONTEXT:\n{context}\n\nRULES:\n{rule_text}\n\n"
            f"COMPUTED FACTS:\n{chr(10).join(facts) if facts else 'none'}\n\n"
            f"CUSTOMER QUESTION:\n{question}"
        )
        reply = chat(SYSTEM, [{"role": "user", "content": message}])
        return {"reply": reply, "retrieved": found["used"], "facts": facts}