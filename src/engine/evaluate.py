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

    totals = torch.zeros(6, device=device, dtype=torch.float64)

    false_idx=[]
    all_preds = []
    all_labels = []
    all_types = []
    all_indices = []

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
                label_type_ids=label_type_ids.to(device),
            )

        labels = batch["labels"]
        ans_type = batch["answer_type"]

        idx = batch["idx"]
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, device=labels.device)
        else:
            idx = idx.to(labels.device)

        preds = answer_logits.argmax(dim=-1)

        preds, labels, ans_type, idx = accelerator.gather_for_metrics(
            (preds, labels, ans_type, idx)
        )

        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_types.append(ans_type.detach().cpu())
        all_indices.append(idx.detach().cpu())
    if len(all_preds) == 0:
        return {
            "overall_acc": 0.0,
            "open_acc": 0.0,
            "closed_acc": 0.0,
            "falsed_samples": [],
        }

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    ans_type = torch.cat(all_types, dim=0)
    indices = torch.cat(all_indices, dim=0)

    answerable = labels.ne(-100)
    correct = preds.eq(labels) & answerable

    open_mask = ans_type.eq(0) & answerable
    closed_mask = ans_type.eq(1) & answerable

    wrong_mask = (~correct) & answerable
    false_idx = indices[wrong_mask].tolist()

    def safe_acc(num: torch.Tensor, den: torch.Tensor) -> float:
        den_value = den.item()
        if den_value == 0:
            return 0.0
        return num.item() / den_value

    return {
        "overall_acc": safe_acc(correct.sum(), answerable.sum()),
        "open_acc": safe_acc((correct & open_mask).sum(), open_mask.sum()),
        "closed_acc": safe_acc((correct & closed_mask).sum(), closed_mask.sum()),
        "falsed_samples": false_idx,
    }