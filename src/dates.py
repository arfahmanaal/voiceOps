import csv
import os
from datetime import date, timedelta

from src.config import DATASET_DIR, TODAY

TODAY_DATE = date.fromisoformat(TODAY)

with open(os.path.join(DATASET_DIR, "data", "holidays.csv"), encoding="utf-8", newline="") as handle:
    HOLIDAYS = {date.fromisoformat(row["date"]) for row in csv.DictReader(handle)}

REFUND_DAYS = {
    "upi": (1, 3),
    "card": (5, 7),
    "netbanking": (3, 5),
    "wallet": (0, 0),
    "cod_wallet": (0, 0),
    "cod_bank": (5, 7),
}

RETURN_DAYS = {
    "electronics": 7,
    "fashion": 15,
    "home_kitchen": 10,
    "beauty_sealed": 7,
    "books_media": 10,
}

ELECTRONICS_CUTOFF = date(2026, 8, 1)
FESTIVE_START = date(2026, 10, 1)
FESTIVE_END = date(2026, 10, 15)


def parse_date(value):
    return date.fromisoformat(str(value)[:10])


def is_business_day(day):
    return day.weekday() != 6 and day not in HOLIDAYS


def add_business_days(start, count):
    day = start
    while count > 0:
        day += timedelta(days=1)
        if is_business_day(day):
            count -= 1
    return day


def business_days_between(start, end):
    count = 0
    day = start
    while day < end:
        day += timedelta(days=1)
        if is_business_day(day):
            count += 1
    return count


def refund_window(initiated_on, method):
    low, high = REFUND_DAYS[method]
    start = parse_date(initiated_on)
    return add_business_days(start, low), add_business_days(start, high)


def refund_overdue_days(initiated_on, method, today=TODAY_DATE):
    _, latest = refund_window(initiated_on, method)
    return business_days_between(latest, today)


def return_deadline(category, delivered_on, placed_on):
    if category not in RETURN_DAYS:
        return None
    delivered = parse_date(delivered_on)
    days = RETURN_DAYS[category]
    if category == "electronics" and delivered < ELECTRONICS_CUTOFF:
        days = 10
    if category == "fashion" and FESTIVE_START <= parse_date(placed_on) <= FESTIVE_END:
        days = 30
    return delivered + timedelta(days=days)