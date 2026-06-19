from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from config import TrainConfig
from data.dataloaders import make_loaders
from engine.evaluate import evaluate
from models.registry import build_model
from utils.io import ensure_dir, save_json
from utils.repro import seed_everything

import os
import json
def _split_params_for_optimizer(model):
    head_params = []
    clip_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("clip."):
            clip_params.append(param)
        else:
            head_params.append(param)

    return head_params, clip_params


def train_main(cfg: TrainConfig):
    seed_everything(cfg.seed)

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)

    if accelerator.is_main_process:
        ensure_dir(cfg.output_dir)
        save_json(cfg, Path(cfg.output_dir) / "config.json")

    processor = CLIPProcessor.from_pretrained(cfg.model_name)
    data_bundle = make_loaders(cfg, processor)
    train_loader = data_bundle["train_loader"]
    eval_loader = data_bundle["eval_loader"]
    test_loader = data_bundle["test_loader"]
    label2ans = data_bundle["label2ans"]
    ans2label = data_bundle["ans2label"]
    label_type_ids = data_bundle["label_type_ids"]
    class_weights = data_bundle["class_weights"]
    train_answer_set = data_bundle["train_answer_set"]
    filter_stats = data_bundle["filter_stats"]

    if accelerator.is_main_process:
        print(
            "[count-filter] "
            f"enabled={filter_stats['filter_invalid_count_answers']} | "
            f"max_num={filter_stats['count_answer_max_num']} | "
            f"train {filter_stats['train_original_len']} -> {filter_stats['train_filtered_len']} "
            f"(invalid_count={filter_stats['train_invalid_count_answer_samples']}) | "
            f"val {filter_stats['eval_original_len']} -> {filter_stats['eval_filtered_len']} "
             f"test {filter_stats['test_original_len']} -> {filter_stats['test_filtered_len']} "
            f"(invalid_count={filter_stats['eval_invalid_count_answer_samples']})"
        )

    model = build_model(cfg, num_answers=len(label2ans))

    head_params, clip_params = _split_params_for_optimizer(model)

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": cfg.lr_head},
            {"params": clip_params, "lr": cfg.lr_clip},
        ],
        weight_decay=cfg.weight_decay,
    )

    total_steps = cfg.epochs * math.ceil(len(train_loader) / max(accelerator.num_processes, 1))
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader,test_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        test_loader,
        scheduler,
    )
    if eval_loader is not None:
        eval_loader= accelerator.prepare(eval_loader)

    class_weights = class_weights.to(accelerator.device)
    label_type_ids = label_type_ids.to(accelerator.device)

    best = -1.0
    best_path = Path(cfg.output_dir) / "best_model.pt"
    test_falsed_sample=Counter()
    val_falsed_sample=Counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_meter = 0.0
        step_count = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            answer_logits, type_logits, _ = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            answer_loss = F.cross_entropy(
                input=answer_logits,
                target=batch["labels"],
                weight=class_weights,
                ignore_index=-100,
            )


            type_loss = F.cross_entropy(type_logits, batch["answer_type"])

            loss = answer_loss + cfg.type_loss_weight * type_loss

            accelerator.backward(loss)

            if cfg.grad_clip and cfg.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()
            scheduler.step()

            loss_meter += accelerator.gather_for_metrics(loss.detach()).mean().item()
            step_count += 1

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            accelerator=accelerator,
            label_type_ids=label_type_ids,
            cfg=cfg,
        )
        if eval_loader is not None:
            eval_metrics = evaluate(
                model=model,
                loader=eval_loader,
                accelerator=accelerator,
                label_type_ids=label_type_ids,
                cfg=cfg,
            )
            val_falsed_sample.update(eval_metrics['falsed_samples'])
        test_falsed_sample.update(test_metrics['falsed_samples'])
        
        if accelerator.is_main_process:
            score = test_metrics["overall_acc"]

            if score > best:
                best = score
                unwrapped = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "model": unwrapped.state_dict(),
                        "label2ans": label2ans,
                        "ans2label": ans2label,
                        "label_type_ids": label_type_ids.detach().cpu(),
                        "train_answer_set": sorted(list(train_answer_set)),
                        "answer_vocab_source": cfg.answer_vocab_source,
                        "filter_stats": filter_stats,
                        "config": asdict(cfg),
                    },
                    best_path,
                )
            if eval_loader is not None:
                print(
                    f"Epoch {epoch:03d}/{cfg.epochs} | "
                    f"loss={loss_meter / max(step_count, 1):.4f} | "
                    f"test_overall={test_metrics['overall_acc']:.4f} | "
                    f"test_open={test_metrics['open_acc']:.4f} | "
                    f"test_closed={test_metrics['closed_acc']:.4f} | "
                    f"val_overall={eval_metrics['overall_acc']:.4f} | "
                    f"val_open={eval_metrics['open_acc']:.4f} | "
                    f"val_closed={eval_metrics['closed_acc']:.4f} | "
                    f"test_best={best:.4f}"
                )
            else:
                 print(
                    f"Epoch {epoch:03d}/{cfg.epochs} | "
                    f"loss={loss_meter / max(step_count, 1):.4f} | "
                    f"test_overall={test_metrics['overall_acc']:.4f} | "
                    f"test_open={test_metrics['open_acc']:.4f} | "
                    f"test_closed={test_metrics['closed_acc']:.4f} | "
                    f"test_best={best:.4f}"
                )
        
    accelerator.wait_for_everyone()
    with open(os.path.join(cfg.output_dir, "falsed_test_samples.json"), "w", encoding="utf-8") as f:
        json.dump(dict(test_falsed_sample), f, ensure_ascii=False, indent=2)
    if eval_loader is not None:
        with open(os.path.join(cfg.output_dir, "falsed_val_samples.json"), "w", encoding="utf-8") as f:
            json.dump(dict(val_falsed_sample), f, ensure_ascii=False, indent=2)
    return str(best_path)