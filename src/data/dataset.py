from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset

from data.answer_type import infer_answer_type
from utils.counting import (
    is_count_question,
    is_invalid_count_answer_sample,
    is_valid_count_answer,
)
from utils.text import clean_question, normalize_answer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MedVQADataset(Dataset):
    """
    Dataset class chung cho các dataset HuggingFace có field:
    - image
    - question
    - answer

    Output vẫn để image/question dạng raw để collate_fn dùng CLIPProcessor xử lý theo batch.
    """
    def __init__(
        self,
        hf_split,
        ans2label: Dict[str, int],
        is_train: bool,
        train_answer_set: Optional[set] = None,
        filter_unseen_train_answers: bool = False,
        filter_invalid_count_answers: bool = False,
        count_answer_max_num: Optional[int] = None,
    ):
        self.ans2label = ans2label
        self.is_train = is_train
        self.train_answer_set = train_answer_set or set(ans2label.keys())
        self.count_answer_max_num = count_answer_max_num

        data = list(hf_split)
        self.original_len = len(data)

        self.invalid_count_answer_samples = [
            x for x in data
            if is_invalid_count_answer_sample(x, max_num=count_answer_max_num)
        ]
        self.num_invalid_count_answer_samples = len(self.invalid_count_answer_samples)

        if filter_invalid_count_answers:
            data = [
                x for x in data
                if not is_invalid_count_answer_sample(x, max_num=count_answer_max_num)
            ]

        if (not is_train) and filter_unseen_train_answers:
            data = [
                x for x in data
                if normalize_answer(x["answer"]) in self.train_answer_set
            ]

        self.data = data
        self.filtered_len = len(self.data)
        self.num_filtered_samples = self.original_len - self.filtered_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        image = sample["image"].convert("RGB")
        question = clean_question(sample["question"])
        answer = normalize_answer(sample["answer"])

        label = self.ans2label.get(answer, -100)
        answer_type = infer_answer_type(sample, answer)
        type_id = 1 if answer_type == "closed" else 0

        seen_in_train = answer in self.train_answer_set
        count_question = is_count_question(question)
        valid_count_answer = is_valid_count_answer(answer, max_num=self.count_answer_max_num)
        invalid_count_answer = count_question and not valid_count_answer

        return {
            "idx":torch.tensor(idx, dtype=torch.long),
            "image": image,
            "question": question,
            "answer": answer,
            "label": torch.tensor(label, dtype=torch.long),
            "answer_type": torch.tensor(type_id, dtype=torch.long),
            "is_unseen_answer": torch.tensor(label == -100, dtype=torch.bool),
            "is_answer_seen_in_train": torch.tensor(seen_in_train, dtype=torch.bool),
            "is_count_question": torch.tensor(count_question, dtype=torch.bool),
            "is_invalid_count_answer": torch.tensor(invalid_count_answer, dtype=torch.bool),
        }
