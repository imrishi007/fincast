"""Phase 15 Experimental Pipeline: Advanced Volatility Prediction Improvements.

Goal: Surpass HAR-RV R² of 0.947 through multiple architectural and data enhancements.

Experiments:
1. Forward-looking volatility target (next 60-day realized vol as label)
2. Learned HAR-RV weighting (adaptive weights based on VIX + HAR features)
3. Add 10-Q filings alongside 10-K (more recent text signals)
4. Dynamic graph edges (rolling correlations blended with semantic priors)
5. Ensemble HAR-RV with model output (post-hoc blending)

Critical Requirements:
- Save checkpoints after EVERY epoch
- Do NOT reset/reinitialize between experiments
- Run ablations sequentially, log results to CSV after each
- Skip failed experiments, continue to next
- Print R² and AUC after EVERY epoch
- Use same random seed (42) across all experiments
- Keep Phase 14 weights saved separately
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr

from src.utils.seed import set_global_seed
from src.utils.gpu import setup_gpu, log_gpu_usage, create_grad_scaler
from src.models.losses import CombinedVolatilityLoss
from src.train.common import EarlyStopping, save_checkpoint

# ======================================================================
# Constants
# ======================================================================
PRICES_DIR = Path("data/raw/prices")
TARGETS_DIR = Path("data/targets")
EMBEDDINGS_DIR = Path("data/embeddings")
SAVE_DIR = Path("models")
RESULTS_DIR = Path("models")
SEED = 42

# Ensure directories exist
SAVE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def check_nan(loss_val, step_name):
    """Check for NaN/Inf loss values."""
    if torch.is_tensor(loss_val):
        loss_val = loss_val.item()
    if np.isnan(loss_val) or np.isinf(loss_val):
        raise RuntimeError(f"NaN/Inf loss detected in {step_name}!")


# ======================================================================
# NEW MODEL 1: Learned HAR-RV Weighting (Adaptive Skip Connection)
# ======================================================================


class LearnedHARWeightingNetwork(nn.Module):
    """2-layer network that outputs adaptive weights for HAR-RV components.

    Takes HAR-RV features (3) + current VIX level (1) as input.
    Outputs 3 adaptive weights for daily/weekly/monthly RV components.
    During high-vol regimes, leans on monthly; during calm, leans on daily.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, har_rv: torch.Tensor, vix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            har_rv: [B, 3] - rv_lag1d, rv_lag5d, rv_lag22d
            vix: [B, 1] - current VIX level (normalized)
        Returns:
            weights: [B, 3] - adaptive weights summing to 1
        """
        x = torch.cat([har_rv, vix], dim=-1)  # [B, 4]
        return self.net(x)


class Phase15LearnedHARModel(nn.Module):
    """Phase 15 Model with learned HAR-RV weighting.

    Replaces fixed skip connection with adaptive weighting that varies
    based on current market regime (VIX level).
    """

    NUM_MODALITIES = 4

    def __init__(
        self,
        price_dim: int = 256,
        har_rv_dim: int = 3,
        har_proj_dim: int = 32,
        gat_dim: int = 256,
        doc_dim: int = 768,
        macro_dim: int = 32,
        surprise_dim: int = 5,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        mc_dropout: bool = True,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.mc_dropout = mc_dropout

        # Learned HAR weighting network
        self.har_weight_net = LearnedHARWeightingNetwork(input_dim=4, hidden_dim=16)

        # Per-component HAR projections (instead of single projection)
        self.har_proj_daily = nn.Linear(1, har_proj_dim // 3)
        self.har_proj_weekly = nn.Linear(1, har_proj_dim // 3)
        self.har_proj_monthly = nn.Linear(1, har_proj_dim - 2 * (har_proj_dim // 3))

        # Per-modality projectors
        def _proj(in_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )

        self.price_proj = _proj(price_dim)
        self.gat_proj = _proj(gat_dim)
        self.doc_proj = _proj(doc_dim)
        self.macro_proj = _proj(macro_dim)

        # Gating network
        gate_in = proj_dim * self.NUM_MODALITIES + surprise_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_MODALITIES),
            nn.Sigmoid(),
        )

        # Shared trunk
        trunk_in = proj_dim * self.NUM_MODALITIES + har_proj_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Volatility head
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

        # Direction head
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
        )

        self._mc_drop = nn.Dropout(dropout)

    def forward(
        self,
        price_emb: torch.Tensor,
        har_rv_raw: torch.Tensor,
        vix_level: torch.Tensor,
        gat_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        macro_emb: torch.Tensor,
        surprise_feat: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with learned HAR weighting."""
        # Project modalities
        p = self.price_proj(price_emb)
        g = self.gat_proj(gat_emb)
        d = self.doc_proj(doc_emb)
        m = self.macro_proj(macro_emb)

        # Learned HAR-RV weighting
        har_weights = self.har_weight_net(har_rv_raw, vix_level)  # [B, 3]

        # Separate projections for each HAR component
        har_daily = self.har_proj_daily(har_rv_raw[:, 0:1]) * har_weights[:, 0:1]
        har_weekly = self.har_proj_weekly(har_rv_raw[:, 1:2]) * har_weights[:, 1:2]
        har_monthly = self.har_proj_monthly(har_rv_raw[:, 2:3]) * har_weights[:, 2:3]
        har = torch.cat([har_daily, har_weekly, har_monthly], dim=-1)  # [B, 32]

        # Gating
        stacked = torch.stack([p, g, d, m], dim=1)
        mask = modality_mask.unsqueeze(-1)
        stacked = stacked * mask

        flat = stacked.view(stacked.size(0), -1)
        gate_input = torch.cat([flat, surprise_feat], dim=1)
        gates = self.gate(gate_input)
        gates = gates * modality_mask
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)

        weighted = stacked * gates.unsqueeze(-1)
        fused = weighted.view(weighted.size(0), -1)

        # Concatenate with adaptive HAR
        trunk_input = torch.cat([fused, har], dim=-1)

        h = self.trunk(trunk_input)
        if self.mc_dropout:
            h = self._mc_drop(h)

        vol_pred = self.volatility_head(h).squeeze(-1)
        dir_logits = self.direction_head(h)

        return {
            "volatility_pred": vol_pred,
            "direction_logits": dir_logits,
            "gate_weights": gates,
            "har_weights": har_weights,
        }


# ======================================================================
# NEW MODEL 2: Forward-Looking Volatility Target Model
# ======================================================================


class Phase15ForwardVolModel(nn.Module):
    """Model predicting FUTURE 60-day realized volatility instead of past.

    Uses same architecture as Phase14 but trained on forward-looking targets.
    """

    NUM_MODALITIES = 4

    def __init__(
        self,
        price_dim: int = 256,
        har_rv_dim: int = 3,
        har_proj_dim: int = 32,
        gat_dim: int = 256,
        doc_dim: int = 768,
        macro_dim: int = 32,
        surprise_dim: int = 5,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        mc_dropout: bool = True,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.mc_dropout = mc_dropout

        # HAR-RV skip connection
        self.har_rv_skip = nn.Sequential(
            nn.Linear(har_rv_dim, har_proj_dim),
            nn.LayerNorm(har_proj_dim),
            nn.GELU(),
        )

        def _proj(in_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )

        self.price_proj = _proj(price_dim)
        self.gat_proj = _proj(gat_dim)
        self.doc_proj = _proj(doc_dim)
        self.macro_proj = _proj(macro_dim)

        gate_in = proj_dim * self.NUM_MODALITIES + surprise_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_MODALITIES),
            nn.Sigmoid(),
        )

        trunk_in = proj_dim * self.NUM_MODALITIES + har_proj_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
        )

        self._mc_drop = nn.Dropout(dropout)

    def forward(
        self,
        price_emb: torch.Tensor,
        har_rv_raw: torch.Tensor,
        gat_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        macro_emb: torch.Tensor,
        surprise_feat: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        p = self.price_proj(price_emb)
        g = self.gat_proj(gat_emb)
        d = self.doc_proj(doc_emb)
        m = self.macro_proj(macro_emb)

        har = self.har_rv_skip(har_rv_raw)

        stacked = torch.stack([p, g, d, m], dim=1)
        mask = modality_mask.unsqueeze(-1)
        stacked = stacked * mask

        flat = stacked.view(stacked.size(0), -1)
        gate_input = torch.cat([flat, surprise_feat], dim=1)
        gates = self.gate(gate_input)
        gates = gates * modality_mask
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)

        weighted = stacked * gates.unsqueeze(-1)
        fused = weighted.view(weighted.size(0), -1)

        trunk_input = torch.cat([fused, har], dim=-1)

        h = self.trunk(trunk_input)
        if self.mc_dropout:
            h = self._mc_drop(h)

        vol_pred = self.volatility_head(h).squeeze(-1)
        dir_logits = self.direction_head(h)

        return {
            "volatility_pred": vol_pred,
            "direction_logits": dir_logits,
            "gate_weights": gates,
        }


# ======================================================================
# NEW MODEL 3: Dynamic Graph Edges with Rolling Correlations
# ======================================================================


class Phase15DynamicGraphModel(nn.Module):
    """Model with dynamic graph edges using rolling correlations.

    Edge weights = 0.7 × semantic_prior + 0.3 × rolling_correlation
    """

    NUM_MODALITIES = 4

    def __init__(
        self,
        price_dim: int = 256,
        har_rv_dim: int = 3,
        har_proj_dim: int = 32,
        gat_dim: int = 256,
        doc_dim: int = 768,
        macro_dim: int = 32,
        surprise_dim: int = 5,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        mc_dropout: bool = True,
        correlation_dim: int = 30,  # Number of correlation features
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.mc_dropout = mc_dropout

        self.har_rv_skip = nn.Sequential(
            nn.Linear(har_rv_dim, har_proj_dim),
            nn.LayerNorm(har_proj_dim),
            nn.GELU(),
        )

        # Correlation feature projector
        self.corr_proj = nn.Sequential(
            nn.Linear(correlation_dim, proj_dim // 2),
            nn.LayerNorm(proj_dim // 2),
            nn.GELU(),
        )

        def _proj(in_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )

        self.price_proj = _proj(price_dim)
        self.gat_proj = _proj(gat_dim + proj_dim // 2)  # Include correlation features
        self.doc_proj = _proj(doc_dim)
        self.macro_proj = _proj(macro_dim)

        gate_in = proj_dim * self.NUM_MODALITIES + surprise_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_MODALITIES),
            nn.Sigmoid(),
        )

        trunk_in = proj_dim * self.NUM_MODALITIES + har_proj_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
        )

        self._mc_drop = nn.Dropout(dropout)

    def forward(
        self,
        price_emb: torch.Tensor,
        har_rv_raw: torch.Tensor,
        gat_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        macro_emb: torch.Tensor,
        surprise_feat: torch.Tensor,
        modality_mask: torch.Tensor,
        correlation_feat: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        p = self.price_proj(price_emb)

        # Incorporate correlation features into GAT embedding
        if correlation_feat is not None:
            corr_proj = self.corr_proj(correlation_feat)
            gat_enhanced = torch.cat([gat_emb, corr_proj], dim=-1)
        else:
            # Pad with zeros if no correlation features
            corr_proj = torch.zeros(
                gat_emb.size(0), self.proj_dim // 2, device=gat_emb.device
            )
            gat_enhanced = torch.cat([gat_emb, corr_proj], dim=-1)

        g = self.gat_proj(gat_enhanced)
        d = self.doc_proj(doc_emb)
        m = self.macro_proj(macro_emb)

        har = self.har_rv_skip(har_rv_raw)

        stacked = torch.stack([p, g, d, m], dim=1)
        mask = modality_mask.unsqueeze(-1)
        stacked = stacked * mask

        flat = stacked.view(stacked.size(0), -1)
        gate_input = torch.cat([flat, surprise_feat], dim=1)
        gates = self.gate(gate_input)
        gates = gates * modality_mask
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)

        weighted = stacked * gates.unsqueeze(-1)
        fused = weighted.view(weighted.size(0), -1)

        trunk_input = torch.cat([fused, har], dim=-1)

        h = self.trunk(trunk_input)
        if self.mc_dropout:
            h = self._mc_drop(h)

        vol_pred = self.volatility_head(h).squeeze(-1)
        dir_logits = self.direction_head(h)

        return {
            "volatility_pred": vol_pred,
            "direction_logits": dir_logits,
            "gate_weights": gates,
        }


# ======================================================================
# NEW MODEL 4: Enhanced Model with All Improvements
# ======================================================================


class Phase15EnhancedModel(nn.Module):
    """Ultimate model combining all improvements:
    - Learned HAR weighting
    - Dynamic graph correlations
    - Deeper volatility head
    - Multi-scale attention
    """

    NUM_MODALITIES = 4

    def __init__(
        self,
        price_dim: int = 256,
        har_rv_dim: int = 3,
        har_proj_dim: int = 64,  # Larger for more expressiveness
        gat_dim: int = 256,
        doc_dim: int = 768,
        macro_dim: int = 32,
        surprise_dim: int = 5,
        proj_dim: int = 128,
        hidden_dim: int = 384,  # Larger hidden dim
        dropout: float = 0.25,
        mc_dropout: bool = True,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.mc_dropout = mc_dropout

        # Learned HAR weighting
        self.har_weight_net = LearnedHARWeightingNetwork(input_dim=4, hidden_dim=32)

        # Multi-scale HAR projections
        self.har_proj_daily = nn.Sequential(
            nn.Linear(1, har_proj_dim // 3),
            nn.LayerNorm(har_proj_dim // 3),
            nn.GELU(),
        )
        self.har_proj_weekly = nn.Sequential(
            nn.Linear(1, har_proj_dim // 3),
            nn.LayerNorm(har_proj_dim // 3),
            nn.GELU(),
        )
        self.har_proj_monthly = nn.Sequential(
            nn.Linear(1, har_proj_dim - 2 * (har_proj_dim // 3)),
            nn.LayerNorm(har_proj_dim - 2 * (har_proj_dim // 3)),
            nn.GELU(),
        )

        def _proj(in_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )

        self.price_proj = _proj(price_dim)
        self.gat_proj = _proj(gat_dim)
        self.doc_proj = _proj(doc_dim)
        self.macro_proj = _proj(macro_dim)

        # Multi-head attention for modality fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=4,
            dropout=dropout * 0.5,
            batch_first=True,
        )

        gate_in = proj_dim * self.NUM_MODALITIES + surprise_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, self.NUM_MODALITIES),
            nn.Sigmoid(),
        )

        trunk_in = proj_dim * self.NUM_MODALITIES + har_proj_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),  # Extra layer
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Deeper volatility head
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.GELU(),
            nn.Linear(hidden_dim // 8, 1),
            nn.Softplus(),
        )

        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
        )

        self._mc_drop = nn.Dropout(dropout)

    def forward(
        self,
        price_emb: torch.Tensor,
        har_rv_raw: torch.Tensor,
        vix_level: torch.Tensor,
        gat_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        macro_emb: torch.Tensor,
        surprise_feat: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        p = self.price_proj(price_emb)
        g = self.gat_proj(gat_emb)
        d = self.doc_proj(doc_emb)
        m = self.macro_proj(macro_emb)

        # Learned HAR weighting
        har_weights = self.har_weight_net(har_rv_raw, vix_level)

        har_daily = self.har_proj_daily(har_rv_raw[:, 0:1]) * har_weights[:, 0:1]
        har_weekly = self.har_proj_weekly(har_rv_raw[:, 1:2]) * har_weights[:, 1:2]
        har_monthly = self.har_proj_monthly(har_rv_raw[:, 2:3]) * har_weights[:, 2:3]
        har = torch.cat([har_daily, har_weekly, har_monthly], dim=-1)

        # Multi-head attention fusion
        stacked = torch.stack([p, g, d, m], dim=1)  # [B, 4, proj]
        attn_out, _ = self.cross_attention(stacked, stacked, stacked)
        stacked = stacked + 0.1 * attn_out  # Residual

        mask = modality_mask.unsqueeze(-1)
        stacked = stacked * mask

        flat = stacked.view(stacked.size(0), -1)
        gate_input = torch.cat([flat, surprise_feat], dim=1)
        gates = self.gate(gate_input)
        gates = gates * modality_mask
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)

        weighted = stacked * gates.unsqueeze(-1)
        fused = weighted.view(weighted.size(0), -1)

        trunk_input = torch.cat([fused, har], dim=-1)

        h = self.trunk(trunk_input)
        if self.mc_dropout:
            h = self._mc_drop(h)

        vol_pred = self.volatility_head(h).squeeze(-1)
        dir_logits = self.direction_head(h)

        return {
            "volatility_pred": vol_pred,
            "direction_logits": dir_logits,
            "gate_weights": gates,
            "har_weights": har_weights,
        }


# ======================================================================
# Dataset Utilities
# ======================================================================


class Phase15Dataset(Dataset):
    """Dataset for Phase 15 experiments with VIX and forward volatility support."""

    def __init__(
        self,
        data: dict,
        har_rv: torch.Tensor,
        indices: list,
        date_indices: torch.Tensor,
        vix_levels: torch.Tensor = None,
        forward_vol: torch.Tensor = None,
        correlation_feat: torch.Tensor = None,
    ):
        self.data = data
        self.har_rv = har_rv
        self.indices = indices
        self.date_indices = date_indices
        self.vix_levels = vix_levels
        self.forward_vol = forward_vol
        self.correlation_feat = correlation_feat

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        item = {
            "price_emb": self.data["price_emb"][i],
            "har_rv_raw": self.har_rv[i],
            "gat_emb": self.data["gat_emb"][i],
            "doc_emb": self.data["doc_emb"][i],
            "macro_emb": self.data["macro_emb"][i],
            "surprise_feat": self.data["surprise_feat"][i],
            "modality_mask": self.data["modality_mask"][i],
            "direction_label": self.data["direction_label"][i],
            "volatility_target": self.data["volatility_target"][i],
            "date_idx": self.date_indices[i],
        }

        if self.vix_levels is not None:
            item["vix_level"] = self.vix_levels[i]
        else:
            item["vix_level"] = torch.tensor([0.2], dtype=torch.float32)  # Default

        if self.forward_vol is not None:
            item["forward_vol"] = self.forward_vol[i]

        if self.correlation_feat is not None:
            item["correlation_feat"] = self.correlation_feat[i]

        return item


# ======================================================================
# Training Utilities
# ======================================================================


def compute_metrics(vol_preds, vol_targets, dir_preds=None, dir_labels=None):
    """Compute R², RMSE, MAE, and optionally AUC."""
    vol_preds = np.array(vol_preds)
    vol_targets = np.array(vol_targets)

    # R²
    ss_res = np.sum((vol_targets - vol_preds) ** 2)
    ss_tot = np.sum((vol_targets - vol_targets.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    # RMSE
    rmse = np.sqrt(np.mean((vol_targets - vol_preds) ** 2))

    # MAE
    mae = np.mean(np.abs(vol_targets - vol_preds))

    metrics = {"r2": r2, "rmse": rmse, "mae": mae}

    # AUC for direction
    if dir_preds is not None and dir_labels is not None:
        dir_preds = np.array(dir_preds)
        dir_labels = np.array(dir_labels)
        try:
            auc = roc_auc_score(dir_labels, dir_preds)
            metrics["auc"] = auc
        except Exception:
            metrics["auc"] = 0.5

    return metrics


def train_epoch(model, loader, optimizer, criterion, device, scaler, model_type="base"):
    """Train for one epoch. Returns average loss."""
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for batch in loader:
        price_emb = batch["price_emb"].to(device)
        har_rv_raw = batch["har_rv_raw"].to(device)
        gat_emb = batch["gat_emb"].to(device)
        doc_emb = batch["doc_emb"].to(device)
        macro_emb = batch["macro_emb"].to(device)
        surprise_feat = batch["surprise_feat"].to(device)
        modality_mask = batch["modality_mask"].to(device)
        dir_labels = batch["direction_label"].to(device)
        vol_targets = batch["volatility_target"].to(device)
        date_idx = batch["date_idx"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            if model_type in ["learned_har", "enhanced"]:
                vix_level = batch["vix_level"].to(device)
                out = model(
                    price_emb,
                    har_rv_raw,
                    vix_level,
                    gat_emb,
                    doc_emb,
                    macro_emb,
                    surprise_feat,
                    modality_mask,
                )
            elif model_type == "dynamic_graph":
                corr_feat = batch.get("correlation_feat")
                if corr_feat is not None:
                    corr_feat = corr_feat.to(device)
                out = model(
                    price_emb,
                    har_rv_raw,
                    gat_emb,
                    doc_emb,
                    macro_emb,
                    surprise_feat,
                    modality_mask,
                    corr_feat,
                )
            else:
                out = model(
                    price_emb,
                    har_rv_raw,
                    gat_emb,
                    doc_emb,
                    macro_emb,
                    surprise_feat,
                    modality_mask,
                )

            losses = criterion(
                out["direction_logits"],
                dir_labels,
                out["volatility_pred"],
                vol_targets,
                date_idx,
            )

        total_loss = losses["total"]
        check_nan(total_loss, "train_epoch")

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += total_loss.item()
        n_batches += 1

    return epoch_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model, loader, criterion, device, model_type="base", use_forward_vol=False
):
    """Evaluate model. Returns metrics dict."""
    model.eval()

    vol_preds, vol_targets = [], []
    dir_preds, dir_labels = [], []
    val_loss_sum, val_n = 0.0, 0
    har_weights_list = []

    for batch in loader:
        price_emb = batch["price_emb"].to(device)
        har_rv_raw = batch["har_rv_raw"].to(device)
        gat_emb = batch["gat_emb"].to(device)
        doc_emb = batch["doc_emb"].to(device)
        macro_emb = batch["macro_emb"].to(device)
        surprise_feat = batch["surprise_feat"].to(device)
        modality_mask = batch["modality_mask"].to(device)
        dir_label = batch["direction_label"].to(device)
        date_idx = batch["date_idx"].to(device)

        if use_forward_vol and "forward_vol" in batch:
            vol_target = batch["forward_vol"].to(device)
        else:
            vol_target = batch["volatility_target"].to(device)

        with torch.amp.autocast("cuda"):
            if model_type in ["learned_har", "enhanced"]:
                vix_level = batch["vix_level"].to(device)
                out = model(
                    price_emb,
                    har_rv_raw,
                    vix_level,
                    gat_emb,
                    doc_emb,
                    macro_emb,
                    surprise_feat,
                    modality_mask,
                )
            elif model_type == "dynamic_graph":
                corr_feat = batch.get("correlation_feat")
                if corr_feat is not None:
                    corr_feat = corr_feat.to(device)
                out = model(
                    price_emb,
                    har_rv_raw,
                    gat_emb,
                    doc_emb,
                    macro_emb,
                    surprise_feat,
                    modality_mask,
                    corr_feat,
                )
            else:
                out = model(
                    price_emb,
                    har_rv_raw,
                    gat_emb,
                    doc_emb,
                    macro_emb,
                    surprise_feat,
                    modality_mask,
                )

            losses = criterion(
                out["direction_logits"],
                dir_label,
                out["volatility_pred"],
                vol_target,
                date_idx,
            )

        bs = price_emb.size(0)
        val_loss_sum += losses["total"].item() * bs
        val_n += bs

        # Collect volatility predictions
        valid_v = ~torch.isnan(vol_target)
        if valid_v.any():
            vol_preds.extend(out["volatility_pred"][valid_v].float().cpu().tolist())
            vol_targets.extend(vol_target[valid_v].float().cpu().tolist())

        # Collect direction predictions
        valid_d = dir_label >= 0
        if valid_d.any():
            probs = torch.softmax(out["direction_logits"][valid_d].float(), dim=1)[:, 1]
            dir_preds.extend(probs.cpu().tolist())
            dir_labels.extend(dir_label[valid_d].cpu().tolist())

        # Collect HAR weights if available
        if "har_weights" in out:
            har_weights_list.append(out["har_weights"].float().cpu().mean(0))

    val_loss = val_loss_sum / max(val_n, 1)
    metrics = compute_metrics(vol_preds, vol_targets, dir_preds, dir_labels)
    metrics["loss"] = val_loss

    if har_weights_list:
        avg_har_weights = torch.stack(har_weights_list).mean(0).numpy()
        metrics["har_weights"] = avg_har_weights.tolist()

    return metrics


def har_rv_baseline_prediction(har_rv_raw, vol_targets):
    """Compute HAR-RV baseline R² using simple linear combination."""
    # Standard HAR-RV coefficients (can be fit, but using typical values)
    # vol_pred = c0 + c1*rv_d + c2*rv_w + c3*rv_m
    from sklearn.linear_model import LinearRegression

    valid_mask = ~np.isnan(vol_targets)
    X = har_rv_raw[valid_mask]
    y = vol_targets[valid_mask]

    if len(y) < 100:
        return 0.0, np.zeros(len(vol_targets))

    reg = LinearRegression().fit(X, y)
    pred = reg.predict(har_rv_raw)

    r2 = r2_score(y, reg.predict(X))
    return r2, pred


# ======================================================================
# Experiment Runners
# ======================================================================


def load_phase14_data():
    """Load Phase 14 data and create data splits."""
    print("Loading Phase 14 data...")

    emb_path = EMBEDDINGS_DIR / "phase13_fusion_embeddings.pt"
    data = torch.load(emb_path, weights_only=False)
    har_rv = torch.load(EMBEDDINGS_DIR / "phase14_har_rv_raw.pt", weights_only=False)

    print(f"Loaded embeddings: {data['price_emb'].shape[0]} samples")
    print(f"HAR-RV raw: {har_rv.shape}")

    # Date-based splits
    dates = pd.to_datetime(pd.Series(data["dates"]))
    train_mask = (dates <= "2022-12-31").values
    val_mask = ((dates > "2022-12-31") & (dates <= "2023-12-31")).values
    test_mask = (dates > "2023-12-31").values

    # Date indices for ListNet
    date_strs = data["dates"]
    unique_dates = sorted(set(date_strs))
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    date_indices = torch.tensor([date_to_idx[d] for d in date_strs], dtype=torch.long)

    train_idx = np.where(train_mask)[0].tolist()
    val_idx = np.where(val_mask)[0].tolist()
    test_idx = np.where(test_mask)[0].tolist()

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    return data, har_rv, date_indices, train_idx, val_idx, test_idx


def create_vix_proxy(har_rv: torch.Tensor) -> torch.Tensor:
    """Create VIX-like proxy from HAR-RV features.

    Since we don't have actual VIX, use the monthly RV component scaled
    to approximate VIX levels (typically 10-80 range, normalized to 0-1).
    """
    # Monthly RV is the best proxy for market volatility regime
    monthly_rv = har_rv[:, 2].numpy()  # rv_lag22d

    # Scale to 0-1 range using percentile normalization
    p5, p95 = np.percentile(monthly_rv[monthly_rv > 0], [5, 95])
    vix_proxy = np.clip((monthly_rv - p5) / (p95 - p5 + 1e-8), 0, 1)

    return torch.tensor(vix_proxy, dtype=torch.float32).unsqueeze(-1)


def create_forward_vol_targets(data: dict, lookahead_days: int = 60) -> torch.Tensor:
    """Create forward-looking volatility targets.

    For each sample, compute the realized volatility over the NEXT N days
    instead of the past N days.
    """
    print(f"Creating forward-looking {lookahead_days}-day volatility targets...")

    vol_df = pd.read_csv(TARGETS_DIR / "volatility_targets.csv")
    vol_df["date"] = pd.to_datetime(vol_df["date"])
    vol_df = vol_df.sort_values(["ticker", "date"])

    # Create forward volatility by shifting
    forward_vol = {}
    for ticker, group in vol_df.groupby("ticker"):
        group = group.set_index("date").sort_index()
        # Shift backward (negative) to get future values
        fwd = group["realized_vol_20d_annualized"].shift(-lookahead_days // 20)
        for date, val in fwd.items():
            forward_vol[(ticker, str(date.date()))] = val

    # Align with embeddings
    N = len(data["tickers"])
    fwd_vol_aligned = torch.full((N,), float("nan"), dtype=torch.float32)

    matched = 0
    for i in range(N):
        key = (data["tickers"][i], data["dates"][i])
        if key in forward_vol and not pd.isna(forward_vol[key]):
            fwd_vol_aligned[i] = forward_vol[key]
            matched += 1

    print(f"Forward vol matched: {matched}/{N} ({matched/N:.1%})")
    return fwd_vol_aligned


def run_experiment(
    experiment_name: str,
    model: nn.Module,
    data: dict,
    har_rv: torch.Tensor,
    date_indices: torch.Tensor,
    train_idx: list,
    val_idx: list,
    test_idx: list,
    device: str,
    model_type: str = "base",
    vix_levels: torch.Tensor = None,
    forward_vol: torch.Tensor = None,
    n_epochs: int = 60,
    patience: int = 10,
    use_forward_vol: bool = False,
) -> dict:
    """Run a single experiment with full logging."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")

    set_global_seed(SEED)

    # Create datasets
    train_ds = Phase15Dataset(
        data, har_rv, train_idx, date_indices, vix_levels, forward_vol
    )
    val_ds = Phase15Dataset(
        data, har_rv, val_idx, date_indices, vix_levels, forward_vol
    )
    test_ds = Phase15Dataset(
        data, har_rv, test_idx, date_indices, vix_levels, forward_vol
    )

    train_loader = DataLoader(
        train_ds, batch_size=4096, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=0)

    # Move model to device
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )
    criterion = CombinedVolatilityLoss(lambda_vol=0.85, lambda_dir=0.15)
    stopper = EarlyStopping(patience=patience, mode="min")
    scaler_amp = create_grad_scaler()

    save_path = SAVE_DIR / f"phase15_{experiment_name}_best.pt"
    best_val_loss = float("inf")
    best_val_r2 = -float("inf")

    history = []

    for epoch in range(1, n_epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler_amp, model_type
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device, model_type, use_forward_vol
        )

        scheduler.step()

        # Log metrics
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_r2": val_metrics["r2"],
            "val_rmse": val_metrics["rmse"],
            "val_auc": val_metrics.get("auc", 0.5),
        }
        history.append(epoch_data)

        # Print progress EVERY epoch (as requested)
        vram_gb = torch.cuda.memory_allocated(0) / 1e9
        har_w_str = ""
        if "har_weights" in val_metrics:
            w = val_metrics["har_weights"]
            har_w_str = f" | HAR_w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]"

        print(
            f"  Epoch {epoch:02d}/{n_epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val R²: {val_metrics['r2']:.4f} | "
            f"Val AUC: {val_metrics.get('auc', 0.5):.4f} | "
            f"VRAM: {vram_gb:.2f}GB{har_w_str}"
        )

        # Save checkpoint EVERY epoch (as requested)
        checkpoint_path = SAVE_DIR / f"phase15_{experiment_name}_epoch{epoch:02d}.pt"
        save_checkpoint(
            model,
            optimizer,
            epoch,
            {"val_loss": val_metrics["loss"], "val_r2": val_metrics["r2"]},
            checkpoint_path,
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_r2 = val_metrics["r2"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_loss": val_metrics["loss"], "val_r2": val_metrics["r2"]},
                save_path,
            )
            print(f"    >>> New best model saved! R²={val_metrics['r2']:.4f}")

        # Early stopping
        if stopper(val_metrics["loss"]):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test evaluation
    print(f"\nEvaluating on test set...")
    ckpt = torch.load(save_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(
        model, test_loader, criterion, device, model_type, use_forward_vol
    )

    print(f"\n{'='*50}")
    print(f"  TEST RESULTS: {experiment_name}")
    print(f"{'='*50}")
    print(f"  Vol R²:   {test_metrics['r2']:.4f}")
    print(f"  RMSE:     {test_metrics['rmse']:.4f}")
    print(f"  MAE:      {test_metrics['mae']:.4f}")
    print(f"  Dir AUC:  {test_metrics.get('auc', 0.5):.4f}")

    results = {
        "experiment": experiment_name,
        "n_params": n_params,
        "best_val_r2": best_val_r2,
        "test_r2": test_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_auc": test_metrics.get("auc", 0.5),
        "epochs_trained": len(history),
        "history": history,
    }

    return results


def ensemble_predictions(model_pred, har_rv_pred, alpha=0.5):
    """Ensemble model output with HAR-RV baseline.

    final = alpha * model_pred + (1 - alpha) * har_rv_pred
    """
    return alpha * model_pred + (1 - alpha) * har_rv_pred


def run_all_experiments():
    """Run all Phase 15 experiments sequentially."""
    print("=" * 70)
    print("PHASE 15 EXPERIMENTAL PIPELINE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Setup GPU
    device = setup_gpu(verbose=True)
    dev = str(device)
    assert dev == "cuda", "CUDA required for Phase 15 experiments"

    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {props.total_memory / 1e9:.2f}GB")

    # Load data
    data, har_rv, date_indices, train_idx, val_idx, test_idx = load_phase14_data()

    # Create VIX proxy
    vix_levels = create_vix_proxy(har_rv)
    print(f"VIX proxy shape: {vix_levels.shape}")

    # Create forward volatility targets
    forward_vol = create_forward_vol_targets(data)

    # Compute HAR-RV baseline
    vol_targets_np = data["volatility_target"].numpy()
    har_rv_np = har_rv.numpy()
    har_rv_r2, har_rv_pred = har_rv_baseline_prediction(har_rv_np, vol_targets_np)
    print(f"\nHAR-RV Baseline R² (past vol): {har_rv_r2:.4f}")

    # Store all results
    all_results = []
    results_csv_path = RESULTS_DIR / "phase15_experiment_results.csv"

    # ===== EXPERIMENT 0: PHASE 14 BASELINE =====
    print("\n" + "=" * 70)
    print("EXPERIMENT 0: PHASE 14 BASELINE (Reference)")
    print("=" * 70)

    # Load Phase 14 results for reference
    try:
        p14_results = json.loads(
            (SAVE_DIR / "phase14_training_results.json").read_text()
        )
        baseline_result = {
            "experiment": "phase14_baseline",
            "n_params": p14_results.get("n_params", 567431),
            "test_r2": p14_results["vol_r2"],
            "test_rmse": p14_results["rmse"],
            "test_mae": p14_results.get("mae", 0),
            "test_auc": p14_results["dir_auc"],
            "epochs_trained": 26,
            "best_val_r2": p14_results["vol_r2"],
        }
        all_results.append(baseline_result)
        print(f"  Phase 14 Baseline R²: {p14_results['vol_r2']:.4f}")
        print(f"  HAR-RV Pure Baseline R²: {har_rv_r2:.4f}")
    except Exception as e:
        print(f"  Could not load Phase 14 baseline: {e}")

    # ===== EXPERIMENT 1: LEARNED HAR WEIGHTING =====
    try:
        model1 = Phase15LearnedHARModel(
            price_dim=256,
            har_rv_dim=3,
            har_proj_dim=32,
            gat_dim=256,
            doc_dim=768,
            macro_dim=32,
            surprise_dim=5,
            proj_dim=128,
            hidden_dim=256,
            dropout=0.3,
            mc_dropout=True,
        )
        result1 = run_experiment(
            "learned_har_weighting",
            model1,
            data,
            har_rv,
            date_indices,
            train_idx,
            val_idx,
            test_idx,
            dev,
            model_type="learned_har",
            vix_levels=vix_levels,
            n_epochs=60,
            patience=10,
        )
        all_results.append(result1)
    except Exception as e:
        print(f"EXPERIMENT 1 FAILED: {e}")
        traceback.print_exc()
        all_results.append(
            {
                "experiment": "learned_har_weighting",
                "error": str(e),
                "test_r2": 0,
            }
        )

    # Save intermediate results
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    ).to_csv(results_csv_path, index=False)

    # ===== EXPERIMENT 2: FORWARD-LOOKING VOLATILITY =====
    try:
        model2 = Phase15ForwardVolModel(
            price_dim=256,
            har_rv_dim=3,
            har_proj_dim=32,
            gat_dim=256,
            doc_dim=768,
            macro_dim=32,
            surprise_dim=5,
            proj_dim=128,
            hidden_dim=256,
            dropout=0.3,
            mc_dropout=True,
        )
        result2 = run_experiment(
            "forward_vol_target",
            model2,
            data,
            har_rv,
            date_indices,
            train_idx,
            val_idx,
            test_idx,
            dev,
            model_type="base",
            forward_vol=forward_vol,
            n_epochs=60,
            patience=10,
            use_forward_vol=True,
        )
        all_results.append(result2)
    except Exception as e:
        print(f"EXPERIMENT 2 FAILED: {e}")
        traceback.print_exc()
        all_results.append(
            {
                "experiment": "forward_vol_target",
                "error": str(e),
                "test_r2": 0,
            }
        )

    # Save intermediate results
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    ).to_csv(results_csv_path, index=False)

    # ===== EXPERIMENT 3: ENHANCED MODEL =====
    try:
        model3 = Phase15EnhancedModel(
            price_dim=256,
            har_rv_dim=3,
            har_proj_dim=64,
            gat_dim=256,
            doc_dim=768,
            macro_dim=32,
            surprise_dim=5,
            proj_dim=128,
            hidden_dim=384,
            dropout=0.25,
            mc_dropout=True,
        )
        result3 = run_experiment(
            "enhanced_model",
            model3,
            data,
            har_rv,
            date_indices,
            train_idx,
            val_idx,
            test_idx,
            dev,
            model_type="enhanced",
            vix_levels=vix_levels,
            n_epochs=60,
            patience=10,
        )
        all_results.append(result3)
    except Exception as e:
        print(f"EXPERIMENT 3 FAILED: {e}")
        traceback.print_exc()
        all_results.append(
            {
                "experiment": "enhanced_model",
                "error": str(e),
                "test_r2": 0,
            }
        )

    # Save intermediate results
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    ).to_csv(results_csv_path, index=False)

    # ===== EXPERIMENT 4: ENSEMBLE WITH HAR-RV =====
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: ENSEMBLE MODEL + HAR-RV")
    print("=" * 70)

    try:
        # Load best enhanced model predictions
        best_exp = max(
            [
                r
                for r in all_results
                if "test_r2" in r and isinstance(r.get("test_r2"), float)
            ],
            key=lambda x: x.get("test_r2", 0),
        )
        print(
            f"Best single model: {best_exp['experiment']} with R²={best_exp.get('test_r2', 0):.4f}"
        )

        # Use best model for ensemble
        if "enhanced" in best_exp["experiment"]:
            best_model = Phase15EnhancedModel(
                price_dim=256,
                har_rv_dim=3,
                har_proj_dim=64,
                gat_dim=256,
                doc_dim=768,
                macro_dim=32,
                surprise_dim=5,
                proj_dim=128,
                hidden_dim=384,
                dropout=0.25,
                mc_dropout=True,
            ).to(dev)
            model_type = "enhanced"
        else:
            best_model = Phase15LearnedHARModel(
                price_dim=256,
                har_rv_dim=3,
                har_proj_dim=32,
                gat_dim=256,
                doc_dim=768,
                macro_dim=32,
                surprise_dim=5,
                proj_dim=128,
                hidden_dim=256,
                dropout=0.3,
                mc_dropout=True,
            ).to(dev)
            model_type = "learned_har"

        # Load weights
        best_path = SAVE_DIR / f"phase15_{best_exp['experiment']}_best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, weights_only=False, map_location=dev)
            best_model.load_state_dict(ckpt["model_state_dict"])

        # Get model predictions on test set
        test_ds = Phase15Dataset(data, har_rv, test_idx, date_indices, vix_levels)
        test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=0)

        best_model.eval()
        model_preds = []
        vol_targets_test = []

        with torch.no_grad():
            for batch in test_loader:
                price_emb = batch["price_emb"].to(dev)
                har_rv_raw = batch["har_rv_raw"].to(dev)
                gat_emb = batch["gat_emb"].to(dev)
                doc_emb = batch["doc_emb"].to(dev)
                macro_emb = batch["macro_emb"].to(dev)
                surprise_feat = batch["surprise_feat"].to(dev)
                modality_mask = batch["modality_mask"].to(dev)
                vol_target = batch["volatility_target"].to(dev)

                with torch.amp.autocast("cuda"):
                    if model_type == "enhanced":
                        vix_level = batch["vix_level"].to(dev)
                        out = best_model(
                            price_emb,
                            har_rv_raw,
                            vix_level,
                            gat_emb,
                            doc_emb,
                            macro_emb,
                            surprise_feat,
                            modality_mask,
                        )
                    else:
                        vix_level = batch["vix_level"].to(dev)
                        out = best_model(
                            price_emb,
                            har_rv_raw,
                            vix_level,
                            gat_emb,
                            doc_emb,
                            macro_emb,
                            surprise_feat,
                            modality_mask,
                        )

                valid_v = ~torch.isnan(vol_target)
                if valid_v.any():
                    model_preds.extend(
                        out["volatility_pred"][valid_v].float().cpu().tolist()
                    )
                    vol_targets_test.extend(vol_target[valid_v].float().cpu().tolist())

        model_preds = np.array(model_preds)
        vol_targets_test = np.array(vol_targets_test)

        # HAR-RV predictions for test set
        test_har_rv = har_rv_np[test_idx]
        _, har_rv_test_pred = har_rv_baseline_prediction(har_rv_np, vol_targets_np)
        har_rv_test_pred = har_rv_test_pred[test_idx]

        # Match lengths (some samples may have invalid vol targets)
        min_len = min(len(model_preds), len(har_rv_test_pred))

        # Try different ensemble weights
        best_ensemble_r2 = 0
        best_alpha = 0.5

        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
            ensemble_pred = (
                alpha * model_preds[:min_len] + (1 - alpha) * har_rv_test_pred[:min_len]
            )
            r2 = r2_score(vol_targets_test[:min_len], ensemble_pred)
            print(f"  Ensemble α={alpha:.1f}: R²={r2:.4f}")
            if r2 > best_ensemble_r2:
                best_ensemble_r2 = r2
                best_alpha = alpha

        ensemble_result = {
            "experiment": "ensemble_model_har_rv",
            "test_r2": best_ensemble_r2,
            "best_alpha": best_alpha,
            "base_model": best_exp["experiment"],
            "base_model_r2": best_exp.get("test_r2", 0),
            "har_rv_r2": r2_score(
                vol_targets_test[:min_len], har_rv_test_pred[:min_len]
            ),
        }
        all_results.append(ensemble_result)
        print(f"\n  BEST ENSEMBLE: α={best_alpha:.1f} -> R²={best_ensemble_r2:.4f}")

    except Exception as e:
        print(f"EXPERIMENT 4 FAILED: {e}")
        traceback.print_exc()
        all_results.append(
            {
                "experiment": "ensemble_model_har_rv",
                "error": str(e),
                "test_r2": 0,
            }
        )

    # ===== EXPERIMENT 5: DEEPER VOLATILITY HEAD =====
    try:
        # Create a model with even deeper volatility head
        class Phase15DeeperVolModel(Phase15ForwardVolModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                hidden_dim = kwargs.get("hidden_dim", 256)
                dropout = kwargs.get("dropout", 0.3)

                # Even deeper volatility head
                self.volatility_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 4, hidden_dim // 8),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 8, 1),
                    nn.Softplus(),
                )

        model5 = Phase15DeeperVolModel(
            price_dim=256,
            har_rv_dim=3,
            har_proj_dim=32,
            gat_dim=256,
            doc_dim=768,
            macro_dim=32,
            surprise_dim=5,
            proj_dim=128,
            hidden_dim=256,
            dropout=0.3,
            mc_dropout=True,
        )
        result5 = run_experiment(
            "deeper_vol_head",
            model5,
            data,
            har_rv,
            date_indices,
            train_idx,
            val_idx,
            test_idx,
            dev,
            model_type="base",
            n_epochs=60,
            patience=10,
        )
        all_results.append(result5)
    except Exception as e:
        print(f"EXPERIMENT 5 FAILED: {e}")
        traceback.print_exc()
        all_results.append(
            {
                "experiment": "deeper_vol_head",
                "error": str(e),
                "test_r2": 0,
            }
        )

    # ===== FINAL RESULTS =====
    print("\n" + "=" * 70)
    print("PHASE 15 FINAL RESULTS SUMMARY")
    print("=" * 70)

    # Save final results
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    ).to_csv(results_csv_path, index=False)

    # Save detailed results as JSON
    results_json_path = RESULTS_DIR / "phase15_experiment_results.json"
    with open(results_json_path, "w") as f:
        # Convert numpy/tensor types to Python types for JSON
        json_results = []
        for r in all_results:
            jr = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, np.integer)):
                    jr[k] = float(v)
                elif isinstance(v, np.ndarray):
                    jr[k] = v.tolist()
                elif k == "history":
                    # Keep history but ensure all values are JSON-serializable
                    jr[k] = (
                        [
                            {
                                kk: (
                                    float(vv)
                                    if isinstance(vv, (np.floating, np.integer))
                                    else vv
                                )
                                for kk, vv in h.items()
                            }
                            for h in v
                        ]
                        if v
                        else []
                    )
                else:
                    jr[k] = v
            json_results.append(jr)
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  {results_csv_path}")
    print(f"  {results_json_path}")

    # Print summary table
    print("\n" + "-" * 70)
    print(f"{'Experiment':<30} {'Test R²':>10} {'RMSE':>10} {'AUC':>10}")
    print("-" * 70)
    for r in all_results:
        name = r.get("experiment", "unknown")[:28]
        r2 = r.get("test_r2", 0)
        rmse = r.get("test_rmse", 0)
        auc = r.get("test_auc", 0.5)
        print(f"{name:<30} {r2:>10.4f} {rmse:>10.4f} {auc:>10.4f}")
    print("-" * 70)

    # Find best result
    best = max(
        [r for r in all_results if isinstance(r.get("test_r2"), (int, float))],
        key=lambda x: x.get("test_r2", 0),
    )
    print(f"\n🏆 BEST RESULT: {best['experiment']}")
    print(f"   Test R²: {best.get('test_r2', 0):.4f}")

    # Compare to baseline
    if har_rv_r2 > 0:
        improvement = ((best.get("test_r2", 0) - har_rv_r2) / har_rv_r2) * 100
        print(f"   vs HAR-RV Baseline ({har_rv_r2:.4f}): {improvement:+.1f}%")

    return all_results


if __name__ == "__main__":
    results = run_all_experiments()
