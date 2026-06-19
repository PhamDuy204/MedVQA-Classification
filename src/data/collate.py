from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import CLIPProcessor


def make_collate_fn(processor: CLIPProcessor, max_length: int = 77):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [x["image"] for x in batch]
        questions = [x["question"] for x in batch]

        enc = processor(
            text=questions,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        enc["idx"] = torch.stack([x["idx"] for x in batch])
        enc["labels"] = torch.stack([x["label"] for x in batch])
        enc["answer_type"] = torch.stack([x["answer_type"] for x in batch])
        enc["is_unseen_answer"] = torch.stack([x["is_unseen_answer"] for x in batch])
        enc["is_answer_seen_in_train"] = torch.stack([x["is_answer_seen_in_train"] for x in batch])
        enc["is_count_question"] = torch.stack([x["is_count_question"] for x in batch])
        enc["is_invalid_count_answer"] = torch.stack([x["is_invalid_count_answer"] for x in batch])
        enc["answers_text"] = [x["answer"] for x in batch]
        enc["questions_text"] = questions
        return enc

    return collate_fn
