from __future__ import annotations

import re

MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2",
    "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
ARTICLES = {"a", "an", "the"}
PUNCT = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=",
    "+", "\\", "_", ">", "<", "@", "`", ",", "?", "!"
]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")

def process_punctuation(text: str) -> str:
    text = str(text)
    out = text
    for p in PUNCT:
        if (p + " " in text) or (" " + p in text) or re.search(COMMA_STRIP, text):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    # giữ dấu '-' trong vài cụm y khoa như "x-ray", rồi chuẩn hóa sau
    out = PERIOD_STRIP.sub("", out)
    return out


def process_digit_article(text: str) -> str:
    words = str(text).lower().split()
    return " ".join(MANUAL_MAP.get(w, w) for w in words if w not in ARTICLES)


def normalize_answer(answer: str) -> str:
    answer = str(answer).lower().strip()
    answer = answer.replace("? -yes/no", "")
    answer = answer.replace("? -open", "")
    answer = answer.replace("? - open", "")

    # Chuẩn hóa biến thể thường gặp trước khi tách dấu.
    answer = answer.replace("x ray", "xray").replace("x-ray", "xray")
    answer = answer.replace("ct scan", "ct").replace("mri scan", "mri")

    answer = process_punctuation(answer)
    answer = process_digit_article(answer)
    answer = answer.replace("\n", " ")
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip()


def clean_question(question: str) -> str:
    question = str(question).strip()
    question = question.replace("? -yes/no", "")
    question = question.replace("? -open", "")
    question = question.replace("? - open", "")
    question = question.replace("x ray", "x-ray")
    question = re.sub(r"\s+", " ", question)
    return question