from __future__ import annotations

from typing import Any, Dict

from utils.fields import get_field


def infer_answer_type(sample: Dict[str, Any], answer_norm: str) -> str:
    """
    Return "closed" hoặc "open".

    Ưu tiên metadata của dataset nếu có.
    Fallback cuối cùng mới dùng yes/no.
    """
    raw = get_field(sample, ["answer_type", "AnswerType", "answer_type_en", "type"], None)
    if raw is not None:
        raw_s = str(raw).lower()
        if "closed" in raw_s or "close" in raw_s or "yes/no" in raw_s:
            return "closed"
        if "open" in raw_s:
            return "open"

    qraw = get_field(sample, ["question_type", "content_type"], None)
    if qraw is not None:
        qraw_s = str(qraw).lower()
        if "closed" in qraw_s or "yes/no" in qraw_s:
            return "closed"
        if "open" in qraw_s:
            return "open"

    return "closed" if answer_norm in {"yes", "no"} else "open"
