from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from data.answer_type import infer_answer_type
from utils.text import normalize_answer


def build_answer_vocab(
    train_split,
    eval_split=None,
    min_freq: int = 1,
    max_answers: Optional[int] = None,
    source: str = "train_eval",
) -> Tuple[List[str], Dict[str, int], set]:
    """
    source="train":
        Strict protocol. Candidate answers chỉ lấy từ train.
        Eval answer không nằm trong train vẫn nằm trong denominator và tính sai.

    source="train_eval":
        Repo/paper-compatible protocol thường gặp trong Med-VQA nhỏ.
        Candidate answers lấy từ train + eval split cố định.
        Khi báo cáo nên ghi rõ protocol này.
    """
    source = source.lower().strip()
    train_answers = [normalize_answer(x["answer"]) for x in train_split]
    train_answer_set = set(train_answers)

    if source in {"train", "strict"}:
        vocab_answers = train_answers
    elif source in {"train_eval", "all", "repo", "paper"}:
        if eval_split is None:
            raise ValueError("eval_split is required when source='train_eval'.")
        vocab_answers = train_answers + [normalize_answer(x["answer"]) for x in eval_split]
    else:
        raise ValueError(f"Unknown answer vocab source: {source}. Use 'train' or 'train_eval'.")

    counter = Counter(vocab_answers)
    items = [(ans, freq) for ans, freq in counter.items() if freq >= min_freq]
    items = sorted(items, key=lambda x: (-x[1], x[0]))

    if max_answers is not None:
        items = items[:max_answers]

    label2ans = [ans for ans, _ in items]
    ans2label = {ans: i for i, ans in enumerate(label2ans)}
    return label2ans, ans2label, train_answer_set


def build_label_type_ids(train_split, ans2label: Dict[str, int]) -> torch.Tensor:
    votes = defaultdict(Counter)

    for sample in train_split:
        ans = normalize_answer(sample["answer"])
        if ans not in ans2label:
            continue

        ans_type = infer_answer_type(sample, ans)
        votes[ans2label[ans]][ans_type] += 1

    label_type = torch.zeros(len(ans2label), dtype=torch.long)

    for label, counter in votes.items():
        label_type[label] = 1 if counter["closed"] >= counter["open"] else 0

    return label_type


def build_class_weights(train_split, ans2label: Dict[str, int], power: float = 0.5) -> torch.Tensor:
    counter = torch.ones(len(ans2label), dtype=torch.float32)

    # for sample in train_split:
    #     ans = normalize_answer(sample["answer"])
    #     if ans in ans2label:
    #         counter[ans2label[ans]] += 1.0

    weights = 1.0 / torch.pow(counter, power)
    weights = weights / weights.mean()
    return weights
