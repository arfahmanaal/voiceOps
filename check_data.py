from src.dates import refund_window, return_deadline
from src.kb import find_products, load_sections

print(len(load_sections()))
print(refund_window("2026-10-17", "card"))
print(return_deadline("fashion", "2026-10-14", "2026-10-06"))
print([p["sku"] for p in find_products("Is ELE-0012 assured?")])