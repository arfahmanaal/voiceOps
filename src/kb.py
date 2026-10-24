import csv
import glob
import os
import re

from src.config import DATASET_DIR

SECTION = re.compile(r"^## \[([^\]]+)\] (.+)$", re.M)
SKU = re.compile(r"\b[A-Za-z]{3}-\d{4}\b")


def parse_header(text):
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    header = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        header[key.strip()] = value.strip()
    return header, text[match.end():]


def load_sections():
    chunks = []
    for path in sorted(glob.glob(os.path.join(DATASET_DIR, "kb", "*.md"))):
        with open(path, encoding="utf-8") as handle:
            text = handle.read().replace("\r\n", "\n")
        header, body = parse_header(text)
        matches = list(SECTION.finditer(body))
        intro = "\n".join(
            line for line in body[: matches[0].start()].splitlines() if not line.startswith("# ")
        ).strip()
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            section_text = body[match.start():end].strip()
            if intro:
                section_text = intro + "\n" + section_text
            chunks.append(
                {
                    "id": match.group(1),
                    "title": match.group(2),
                    "doc_id": header["doc_id"],
                    "doc_title": header["title"],
                    "doc_type": header["doc_type"],
                    "status": header["status"],
                    "effective_from": header["effective_from"],
                    "language": header["language"],
                    "text": section_text,
                }
            )
    return chunks


def load_products():
    with open(os.path.join(DATASET_DIR, "data", "products.csv"), encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


PRODUCTS = load_products()
PRODUCTS_BY_SKU = {product["sku"].upper(): product for product in PRODUCTS}


def find_products(text):
    lowered = text.lower()
    found = {}
    for sku in SKU.findall(text):
        product = PRODUCTS_BY_SKU.get(sku.upper())
        if product:
            found[product["sku"]] = product
    for product in PRODUCTS:
        title = product["title"].lower()
        if title in lowered or title.split(",")[0] in lowered:
            found[product["sku"]] = product
    return list(found.values())