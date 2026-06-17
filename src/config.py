from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Any, Dict

import yaml


@dataclass
class TrainConfig:
    dataset_name: str = "flaviagiammarino/vqa-rad"
    model_name: str = "flaviagiammarino/pubmed-clip-vit-base-patch32"
    model_type: str = "pubmedclip_ban"

    output_dir: str = "runs/pubmedclip_ban"
    seed: int = 42

    epochs: int = 40
    batch_size: int = 16
    eval_batch_size: int = 64
    num_workers: int = 2
    max_length: int = 77

    num_hid: int = 512
    glimpse: int = 4
    freeze_clip: bool = True

    lr_head: float = 1e-4
    lr_clip: float = 1e-6
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    type_loss_weight: float = 0.2
    class_weight_power: float = 0.5
    mixed_precision: str = "fp16"
    grad_clip: float = 1.0

    min_answer_freq: int = 1
    max_answers: Optional[int] = None
    answer_vocab_source: str = "train_eval"
    filter_eval_unseen_train_answers: bool = False

    filter_invalid_count_answers: bool = False
    count_answer_max_num: Optional[int] = None

    eval_with_type_mask: bool = True


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> TrainConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                data[key] = value

    valid_fields = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    return TrainConfig(**data)
