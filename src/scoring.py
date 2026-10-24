import re

DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twelve": "12",
    "fifteen": "15", "twenty": "20", "thirty": "30",
}
WORDS = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b")


def normalize(text):
    text = str(text).translate(DIGITS).lower().replace(",", "")
    text = WORDS.sub(lambda match: NUMBER_WORDS[match.group(1)], text)
    text = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", text)
    text = re.sub(r"(\d+) of ([a-z]+)", r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip()


def has_fact(reply, fact):
    return any(alt.strip() and normalize(alt) in reply for alt in fact.split(" / "))


def check(reply, must_include, must_not_include):
    text = normalize(reply)
    facts = [fact for fact in must_include.split(" ; ") if fact.strip()]
    banned = [item for item in must_not_include.split(" ; ") if item.strip()]
    include_ok = all(has_fact(text, fact) for fact in facts)
    exclude_ok = not any(normalize(item) in text for item in banned)
    return include_ok, exclude_ok