from __future__ import annotations

from typing import Any, Optional, List, Dict

import torch

from utils.fields import get_field
from utils.text import normalize_answer


def is_count_question(question: str) -> bool:
    """
    True nếu question là dạng hỏi số lượng/counting.
    Giữ cùng tinh thần với BAN/TMMPN: how many, number of, amount of, count of.
    """
    q = str(question).lower().strip()
    return (
        "how many" in q
        or ("number of" in q and "number of the" not in q)
        or "amount of" in q
        or "count of" in q
    )


def answer_filter(
    answer_obj_or_text: Any,
    label2ans: Optional[List[str]] = None,
    max_num: Optional[int] = 10,
) -> bool:
    """
    True nếu answer là số lượng hợp lệ.

    Hỗ trợ:
    - answer text: "2", "two", "2.0"
    - BAN/TMMPN target dict: {"labels": tensor/list}

    max_num:
    - 10: giống BAN counting-only gốc.
    - None: chấp nhận mọi số nguyên không âm, phù hợp hơn khi chỉ muốn lọc answer không phải số.
    """
    def _valid_text(ans_text: Any) -> bool:
        ans = normalize_answer(str(ans_text))

        if ans.isdigit():
            value = int(ans)
            return max_num is None or value <= max_num

        # Chấp nhận dạng "2.0" như một số đếm nguyên.
        try:
            value_float = float(ans)
            if value_float.is_integer() and value_float >= 0:
                value = int(value_float)
                return max_num is None or value <= max_num
        except ValueError:
            pass

        return False

    if isinstance(answer_obj_or_text, dict) and label2ans is not None:
        labels = answer_obj_or_text.get("labels", [])
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().tolist()
        return any(_valid_text(label2ans[int(lab)]) for lab in labels)

    return _valid_text(answer_obj_or_text)


def is_valid_count_answer(answer: Any, max_num: Optional[int] = None) -> bool:
    """True nếu raw answer là số lượng hợp lệ. Mặc định không giới hạn <=10."""
    return answer_filter(answer, label2ans=None, max_num=max_num)


def is_invalid_count_answer_sample(sample: Dict[str, Any], max_num: Optional[int] = None) -> bool:
    """
    True nếu sample nên bị xóa theo rule:
    question hỏi số lượng nhưng answer không phải số lượng hợp lệ.
    """
    q = get_field(sample, ["question", "Question"], "")
    a = get_field(sample, ["answer", "Answer"], "")
    return is_count_question(q) and not is_valid_count_answer(a, max_num=max_num)


def is_howmany(
    question: str,
    answer: Optional[Any] = None,
    label2ans: Optional[List[str]] = None,
    max_num: Optional[int] = 10,
) -> bool:
    """
    Backward-compatible helper giống BAN/TMMPN.
    True khi question là counting và answer là số hợp lệ.
    Nếu answer=None thì chỉ kiểm tra question có phải counting hay không.
    """
    if not is_count_question(question):
        return False
    if answer is None:
        return True
    return answer_filter(answer, label2ans=label2ans, max_num=max_num)