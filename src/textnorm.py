import re

DEVANAGARI = str.maketrans("०१२३४५६७८९", "0123456789")

DIGIT_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "shunya": "0", "ek": "1", "do": "2", "teen": "3", "char": "4", "chaar": "4",
    "paanch": "5", "panch": "5", "chhe": "6", "chhah": "6", "saat": "7", "aath": "8", "nau": "9",
}
AMBIGUOUS = {"to": "2", "too": "2", "for": "4", "oh": "0", "o": "0"}
REPEAT = {"double": 2, "triple": 3}


def core(token):
    return re.sub(r"^\W+|\W+$", "", token.lower())


def normalize_speech(text):
    tokens = text.translate(DEVANAGARI).split()
    kinds = []
    for token in tokens:
        word = core(token)
        if word.isdigit():
            kinds.append("digit")
        elif word in DIGIT_WORDS or word in REPEAT:
            kinds.append("word")
        elif word in AMBIGUOUS:
            kinds.append("maybe")
        else:
            kinds.append("other")
    output = []
    index = 0
    while index < len(tokens):
        if kinds[index] == "other":
            output.append(tokens[index])
            index += 1
            continue
        end = index
        while end < len(tokens) and kinds[end] != "other":
            end += 1
        group = list(range(index, end))
        has_word = any(kinds[i] == "word" for i in group)
        if has_word and len(group) >= 3:
            digits = ""
            position = index
            while position < end:
                word = core(tokens[position])
                if word in REPEAT and position + 1 < end:
                    nxt = core(tokens[position + 1])
                    digit = DIGIT_WORDS.get(nxt) or AMBIGUOUS.get(nxt) or (nxt if nxt.isdigit() else "")
                    digits += digit * REPEAT[word]
                    position += 2
                    continue
                digits += word if word.isdigit() else DIGIT_WORDS.get(word) or AMBIGUOUS.get(word, "")
                position += 1
            output.append(digits)
        else:
            output.extend(tokens[index:end])
        index = end
    return " ".join(output)


def digit_runs(text):
    runs = []
    for match in re.finditer(r"\d(?:[\d\s\-]*\d)?", text):
        digits = re.sub(r"\D", "", match.group())
        if len(digits) == 12:
            runs.extend([digits[:8], digits[8:]])
        elif len(digits) == 14:
            runs.extend([digits[:8], digits[8:]])
        else:
            runs.append(digits)
    return runs


def split_runs(runs):
    order_id = next((run for run in runs if len(run) == 8), None)
    proof = None
    seen_order = order_id is None
    for run in runs:
        if run == order_id and not seen_order:
            seen_order = True
            continue
        if len(run) in (4, 6):
            proof = run
            break
    return order_id, proof