import torch
def mask_logits_by_predicted_type(
    logits: torch.Tensor,
    type_logits: torch.Tensor,
    label_type_ids: torch.Tensor,
    mask_value: float = -1e4,
) -> torch.Tensor:
    pred_type = type_logits.argmax(dim=-1)
    allowed = label_type_ids.unsqueeze(0).to(logits.device) == pred_type.unsqueeze(1)
    return logits.masked_fill(~allowed, mask_value)