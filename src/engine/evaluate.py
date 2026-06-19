from __future__ import annotations

from typing import Dict

import torch
from accelerate import Accelerator

from config import TrainConfig
from models.common import mask_logits_by_predicted_type


@torch.no_grad()
def evaluate(
    model,
    loader,
    accelerator: Accelerator,
    label_type_ids: torch.Tensor,
    cfg: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    device = accelerator.device
    label_type_ids = label_type_ids.to(device)

    # [correct_all, total_all, correct_open, total_open, correct_closed, total_closed]
    totals = torch.zeros(6, device=device, dtype=torch.float64)

    false_idx = []

    for batch in loader:
        answer_logits, type_logits, _ = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        if cfg.eval_with_type_mask:
            answer_logits = mask_logits_by_predicted_type(
                logits=answer_logits,
                type_logits=type_logits,
                label_type_ids=label_type_ids,
            )

        labels = batch["labels"]
        ans_type = batch["answer_type"]
        preds = answer_logits.argmax(dim=-1)

        idx = batch["idx"]
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, device=labels.device)
        else:
            idx = idx.to(labels.device)

        # Multi-GPU safe:
        # gather preds/labels/types/idx từ tất cả process
        # và drop duplicate ở batch cuối nếu có.
        preds, labels, ans_type, idx = accelerator.gather_for_metrics(
            (preds, labels, ans_type, idx)
        )

        answerable = labels.ne(-100)
        correct = preds.eq(labels) & answerable

        open_mask = ans_type.eq(0) & answerable
        closed_mask = ans_type.eq(1) & answerable

        totals[0] += correct.sum()
        totals[1] += answerable.sum()

        totals[2] += (correct & open_mask).sum()
        totals[3] += open_mask.sum()

        totals[4] += (correct & closed_mask).sum()
        totals[5] += closed_mask.sum()

        # Vì sau gather, mọi process đều có cùng wrong_idx.
        # Chỉ main process giữ list này để tránh duplicate.
        if accelerator.is_main_process:
            wrong_mask = (~correct) & answerable
            false_idx.extend(idx[wrong_mask].detach().cpu().tolist())

        del answer_logits, type_logits, preds, labels, ans_type, idx

    def safe_div(a: torch.Tensor, b: torch.Tensor) -> float:
        b = b.item()
        if b == 0:
            return 0.0
        return a.item() / b

    metrics = {
        "overall_acc": safe_div(totals[0], totals[1]),
        "open_acc": safe_div(totals[2], totals[3]),
        "closed_acc": safe_div(totals[4], totals[5]),
        "falsed_samples": false_idx if accelerator.is_main_process else [],
    }

    return metrics
