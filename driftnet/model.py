"""DriftNet Dual-Head Trajectory Transformer (Section V-B of Pinjari & Saint-Germain, 2026).

Architecture specifications from paper:
- Input dim: 772 (768 frozen all-mpnet-base-v2 + 4 identity-free world features)
- Linear projection: 772 -> d_model (256)
- Sinusoidal positional encoding (max_len = 64) + Dropout
- Transformer encoder: L layers (default L=3 selected config, or L=2 reference),
  nhead = 4, dim_feedforward = 512, activation = GELU, padding-masked self-attention
- Head 1 (Trajectory): masked mean pool over valid steps -> Linear(d, d/2) -> GELU -> Dropout -> Linear(d/2, 1)
- Head 2 (Steps): per-step Linear(d, d/2) -> GELU -> Dropout -> Linear(d/2, 4)
- Trainable parameters: 1,318,533 (L=2) / 1,845,637 (L=3)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import IDX_TO_LABEL


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Eq. 8 in paper)."""

    def __init__(self, d_model: int = 256, max_len: int = 64, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class DriftNet(nn.Module):
    """Dual-head trajectory Transformer for joint trajectory detection and 4-way step labeling."""

    def __init__(
        self,
        input_dim: int = 772,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        num_step_classes: int = 4,
        max_len: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers

        # Eq. 7: Learned linear projection W_p x_t + b_p
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # Eq. 9: Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Eq. 10: Head 1 (Trajectory compromise probability)
        self.mlp_traj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Eq. 11: Head 2 (Per-step 4-way classification)
        self.mlp_step = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model // 2, num_step_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, T, 772) concatenated step embeddings and world features.
            valid_mask: (B, T) boolean or float mask where 1/True = real step, 0/False = padding.

        Returns:
            traj_logits: (B,) raw logits for trajectory compromise.
            step_logits: (B, T, 4) raw logits for per-step 4-way labels.
        """
        key_padding_mask = ~valid_mask.bool()  # True indicates padded positions to ignore
        h0 = self.input_proj(x)
        h0 = self.pos_enc(h0)
        hL = self.encoder(h0, src_key_padding_mask=key_padding_mask)

        # Eq. 10: Masked mean pooling over valid steps
        mask_float = valid_mask.float().unsqueeze(-1)  # (B, T, 1)
        pooled = (hL * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
        traj_logits = self.mlp_traj(pooled).squeeze(-1)  # (B,)

        # Eq. 11: Per-step scoring on shared states
        step_logits = self.mlp_step(hL)  # (B, T, 4)
        return traj_logits, step_logits

    @torch.no_grad()
    def predict_prefix_stream(
        self,
        step_embeddings: np.ndarray,
        world_features: np.ndarray,
        device: str = "cpu",
    ) -> List[Dict]:
        """Score every growing prefix x_{1:1}, x_{1:2}, ..., x_{1:T} in one batched forward pass!

        Because a T-step trajectory has T prefixes, we construct a batch of size T where row k
        contains steps 1..k (and padding for k+1..T), allowing all T streaming prefixes to be
        evaluated in a single GPU/CPU forward pass (< 2ms).
        """
        self.eval()
        T = step_embeddings.shape[0]
        x_full = np.concatenate([step_embeddings, world_features], axis=-1).astype(np.float32)  # (T, 772)

        # Build batch of T prefixes: shape (T, T, 772)
        batch_x = np.zeros((T, T, self.input_dim), dtype=np.float32)
        batch_mask = np.zeros((T, T), dtype=bool)
        for k in range(T):
            batch_x[k, : k + 1, :] = x_full[: k + 1, :]
            batch_mask[k, : k + 1] = True

        x_tensor = torch.from_numpy(batch_x).to(device)
        mask_tensor = torch.from_numpy(batch_mask).to(device)

        traj_logits, step_logits = self.forward(x_tensor, mask_tensor)
        traj_probs = torch.sigmoid(traj_logits).cpu().numpy()
        step_probs = F.softmax(step_logits, dim=-1).cpu().numpy()

        prefix_snapshots = []
        for k in range(T):
            curr_len = k + 1
            curr_step_probs = step_probs[k, :curr_len, :]  # (curr_len, 4)
            curr_step_preds = curr_step_probs.argmax(axis=-1)
            step_verdicts = []
            for s_idx in range(curr_len):
                pred_cls = int(curr_step_preds[s_idx])
                step_verdicts.append({
                    "step_index": s_idx,
                    "pred_label": IDX_TO_LABEL[pred_cls],
                    "probabilities": {
                        IDX_TO_LABEL[c]: round(float(curr_step_probs[s_idx, c]), 4)
                        for c in range(4)
                    },
                })

            has_hijacked = any(sv["pred_label"] == "hijacked" for sv in step_verdicts)
            inj_points = [sv["step_index"] for sv in step_verdicts if sv["pred_label"] == "injection_point"]
            hijacked_steps = [sv["step_index"] for sv in step_verdicts if sv["pred_label"] == "hijacked"]
            failed_steps = [sv["step_index"] for sv in step_verdicts if sv["pred_label"] == "failed_injection"]

            traj_prob = float(traj_probs[k])
            is_compromised = traj_prob >= 0.5 or has_hijacked

            prefix_snapshots.append({
                "prefix_length": curr_len,
                "traj_prob": round(traj_prob, 4),
                "traj_compromised": bool(is_compromised),
                "step_verdicts": step_verdicts,
                "injection_points": inj_points,
                "hijacked_steps": hijacked_steps,
                "failed_injection_steps": failed_steps,
            })

        return prefix_snapshots


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
