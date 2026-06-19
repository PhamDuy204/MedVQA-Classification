
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPModel
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F
class FCNet(nn.Module):
    def __init__(self, dims: List[int], act: str = "ReLU", dropout: float = 0.2):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(weight_norm(nn.Linear(dims[i], dims[i + 1],bias=False), name="weight", dim=None))
            if act:
                layers.append(getattr(nn, act)())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class BCNet(nn.Module):
    def __init__(self, v_dim: int, q_dim: int, h_dim: int, h_out: Optional[int] = None,
                 dropout: Tuple[float, float] = (0.2, 0.5), k: int = 3):
        super().__init__()
        self.h_dim = h_dim
        self.h_out = h_out
        self.k = k
        self.v_net = FCNet([v_dim, h_dim * k], act="ReLU", dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * k], act="ReLU", dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])
        self.p_net = nn.AvgPool1d(k, stride=k) if k > 1 else None

        if h_out is not None:
            self.h_mat = nn.Parameter(torch.randn(1, h_out, 1, h_dim * k) * 0.01)
            self.h_bias = nn.Parameter(torch.zeros(1, h_out, 1, 1))

    def forward(self, v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        v_ = self.dropout(self.v_net(v))
        q_ = self.q_net(q)
        if self.h_out is None:
            return torch.einsum("bvk,bqk->bvqk", v_, q_)
        return torch.einsum("ghyk,bvk,bqk->bhvq", self.h_mat, v_, q_) + self.h_bias

    def forward_with_weights(self, v: torch.Tensor, q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        v_ = self.v_net(v)
        q_ = self.q_net(q)
        logits = torch.einsum("bvk,bvq,bqk->bk", v_, w, q_)
        if self.k > 1:
            logits = self.p_net(logits.unsqueeze(1)).squeeze(1) * self.k
        return logits


class BiAttention(nn.Module):
    def __init__(self, v_dim: int, q_dim: int, h_dim: int, glimpse: int = 4):
        super().__init__()
        self.glimpse = glimpse
        self.logits = BCNet(v_dim, q_dim, h_dim, h_out=glimpse, dropout=(0.2, 0.5), k=3)

    def forward_all(self, v: torch.Tensor, q: torch.Tensor, q_mask: Optional[torch.Tensor] = None):
        B, V, Q = v.size(0), v.size(1), q.size(1)
        logits = self.logits(v, q)
        if q_mask is not None:
            mask = q_mask[:, None, None, :].bool()
            logits = logits.masked_fill(~mask, -1e4)
        att = F.softmax(logits.reshape(B, self.glimpse, V * Q), dim=-1).reshape(B, self.glimpse, V, Q)
        return att, logits


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim,bias=False), name="weight", dim=None),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hid_dim, out_dim,bias=False), name="weight", dim=None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class PubMedCLIPBANFixed(nn.Module):
    def __init__(
        self,
        clip_model: CLIPModel,
        num_answers: int,
        num_hid: int = 512,
        glimpse: int = 4,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.clip = clip_model
        self.freeze_clip = freeze_clip
        self.v_dim = clip_model.config.vision_config.hidden_size
        self.q_dim = clip_model.config.text_config.hidden_size
        self.glimpse = glimpse

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.v_att = BiAttention(self.v_dim, self.q_dim, num_hid, glimpse=glimpse)
        self.b_net = nn.ModuleList([
            BCNet(self.v_dim, self.q_dim, num_hid, h_out=None, dropout=(0.2, 0.5), k=3)
            for _ in range(glimpse)
        ])
        self.q_prj = nn.ModuleList([
            FCNet([num_hid, self.q_dim], act="ReLU", dropout=0.2)
            for _ in range(glimpse)
        ])

        self.answer_classifier = SimpleClassifier(self.q_dim, num_hid * 2, num_answers, dropout=0.5)
        self.type_classifier = SimpleClassifier(self.q_dim, num_hid, 2, dropout=0.3)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_clip:
            self.clip.eval()
        return self

    def encode_tokens(self, pixel_values, input_ids, attention_mask):
        grad_enabled = any(p.requires_grad for p in self.clip.parameters())
        with torch.set_grad_enabled(grad_enabled):
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values, return_dict=True)
            text_outputs = self.clip.text_model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
        # ViT: bỏ CLS token; ResNet CLIP có thể cần chỉnh lại nếu dùng RN backend.
        visual_tokens = vision_outputs.last_hidden_state[:, 1:, :]
        text_tokens = text_outputs.last_hidden_state
        return visual_tokens, text_tokens

    def forward(self, pixel_values, input_ids, attention_mask):
        v, q = self.encode_tokens(pixel_values, input_ids, attention_mask)
        att, _ = self.v_att.forward_all(v, q, q_mask=attention_mask)

        for g in range(self.glimpse):
            b_emb = self.b_net[g].forward_with_weights(v, q, att[:, g])
            q = q + self.q_prj[g](b_emb).unsqueeze(1)

        mask = attention_mask.unsqueeze(-1).float()
        q_pooled = (q * mask).sum(1) / mask.sum(1).clamp_min(1.0)

        answer_logits = self.answer_classifier(q_pooled)
        type_logits = self.type_classifier(q_pooled)
        return answer_logits, type_logits, att