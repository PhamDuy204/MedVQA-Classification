from __future__ import annotations

from typing import Any, Dict, Tuple

from data.data_prep import PreDataset
from torch.utils.data import DataLoader,ConcatDataset
from transformers import CLIPProcessor

from config import TrainConfig
from data.collate import make_collate_fn
from data.dataset import MedVQADataset
from data.vocab import build_answer_vocab, build_class_weights, build_label_type_ids


def make_loaders(cfg: TrainConfig, processor: CLIPProcessor):
    train_split = PreDataset(cfg.dataset_name,"train")

    try : eval_split = PreDataset(cfg.dataset_name,"val") 
    except: eval_split=None

    test_split = PreDataset(cfg.dataset_name,"test")


    label2ans, ans2label, train_answer_set = build_answer_vocab(
        train_split=train_split,
        eval_split= test_split if eval_split is None else ConcatDataset([eval_split,test_split]),
        min_freq=cfg.min_answer_freq,
        max_answers=cfg.max_answers,
        source=cfg.answer_vocab_source,
    )

    label_type_ids = build_label_type_ids(train_split, ans2label)
    class_weights = build_class_weights(
        train_split=train_split,
        ans2label=ans2label,
        power=cfg.class_weight_power,
    )

    train_set = MedVQADataset(
        hf_split=train_split,
        ans2label=ans2label,
        is_train=True,
        train_answer_set=train_answer_set,
        filter_invalid_count_answers=cfg.filter_invalid_count_answers,
        count_answer_max_num=cfg.count_answer_max_num,
    )
    
    eval_set = MedVQADataset(
        hf_split=eval_split,
        ans2label=ans2label,
        is_train=False,
        train_answer_set=train_answer_set,
        filter_unseen_train_answers=cfg.filter_eval_unseen_train_answers,
        filter_invalid_count_answers=cfg.filter_invalid_count_answers,
        count_answer_max_num=cfg.count_answer_max_num,
    ) if eval_split is not None else None
    test_set = MedVQADataset(
        hf_split=test_split,
        ans2label=ans2label,
        is_train=False,
        train_answer_set=train_answer_set,
        filter_unseen_train_answers=cfg.filter_eval_unseen_train_answers,
        filter_invalid_count_answers=cfg.filter_invalid_count_answers,
        count_answer_max_num=cfg.count_answer_max_num,
    )

    collate_fn = make_collate_fn(processor, max_length=cfg.max_length)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )  if eval_split is not None else None

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    filter_stats = {
        "train_original_len": train_set.original_len,
        "train_filtered_len": train_set.filtered_len,
        "train_invalid_count_answer_samples": train_set.num_invalid_count_answer_samples,
        "eval_original_len": eval_set.original_len if eval_split is not None else 0,
        "eval_filtered_len": eval_set.filtered_len if eval_split is not None else 0,
        "eval_invalid_count_answer_samples": eval_set.num_invalid_count_answer_samples if eval_split is not None else 0,
        "test_original_len": test_set.original_len ,
        "test_filtered_len": test_set.filtered_len,
        "test_invalid_count_answer_samples": test_set.num_invalid_count_answer_samples,
        "filter_invalid_count_answers": cfg.filter_invalid_count_answers,
        "count_answer_max_num": cfg.count_answer_max_num,
    }

    return {
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "test_loader": test_loader,
        "label2ans": label2ans,
        "ans2label": ans2label,
        "label_type_ids": label_type_ids,
        "class_weights": class_weights,
        "train_answer_set": train_answer_set,
        "filter_stats": filter_stats,
    }
