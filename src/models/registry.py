from __future__ import annotations

from transformers import CLIPModel

from config import TrainConfig
from models.pubmedclip_ban.model import PubMedCLIPBANFixed


def build_model(cfg: TrainConfig, num_answers: int):
    clip_model = CLIPModel.from_pretrained(cfg.model_name)

    if cfg.model_type == "pubmedclip_ban":
        return PubMedCLIPBANFixed(
            clip_model=clip_model,
            num_answers=num_answers,
            num_hid=cfg.num_hid,
            glimpse=cfg.glimpse,
            freeze_clip=cfg.freeze_clip,
        )


    raise ValueError(
        f"Unknown model_type={cfg.model_type}. "
        "Available: pubmedclip_ban, clip_mlp"
    )