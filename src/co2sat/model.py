"""CNN-BiLSTM-Attention model of Mo et al. (2025), Energy 334:137570.

Faithful PyTorch implementation per paper Section 2.2 (Figs. 2-4, eqs. 10-16)
and Supplementary Table S1 (CNN-BiLSTM-Attention row):

  Layer 1   Feature attention: one 16x16 embedding matrix PER time step
            (24 matrices), softmax over bands (eq. 10-12)
  Layer 2   Three parallel Conv1d over time, kernels 3/5/7, padding 1/2/3,
            32 filters each, ReLU
  Layer 3   1x1 Conv1d (96 -> 32), ReLU
  Layer 4-5 Two BiLSTM layers, hidden 64 (output 24 x 128)
  Layer 6   Temporal attention, W_beta in R^{128x1} (eq. 14-16)
  Layer 7   Flatten (3072) + concat 11 static features -> 3083
  Layer 8   Two PARALLEL Linear 3083 -> 1536, ReLU (one per task)
  Layer 9   Two PARALLEL Linear 1536 -> 1, no activation

Dimension check: 24*128 + 11 = 3083, matching Table S1 exactly.

Documented deviations (replication log):
  - Supplementary eq. 13 writes softmax with a negative exponent (e^-x);
    treated as a typographical error, standard softmax adopted.
  - n_bands/n_steps are constructor parameters so the same class serves the
    SEVIRI-subset (12 bands) and FCI-subset ablations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    """Per-time-step band attention (paper eq. 10-12, Fig. 3)."""

    def __init__(self, n_bands: int = 16, n_steps: int = 24) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_steps, n_bands, n_bands))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_bands, n_steps) -> weighted, same shape."""
        xt = x.transpose(1, 2)  # (B, T, bands)
        scores = torch.tanh(torch.einsum("bti,tij->btj", xt, self.W))
        alpha = torch.softmax(scores, dim=-1)
        return (xt * alpha).transpose(1, 2)

    def attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return alpha (B, T, bands) for interpretability figures."""
        xt = x.transpose(1, 2)
        return torch.softmax(
            torch.tanh(torch.einsum("bti,tij->btj", xt, self.W)), dim=-1
        )


class TemporalAttention(nn.Module):
    """Time-step attention over BiLSTM output (paper eq. 14-16, Fig. 4)."""

    def __init__(self, n_features: int = 128) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, 1))
        nn.init.xavier_uniform_(self.W)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, F) -> weighted, same shape (broadcast, eq. 15-16)."""
        beta = torch.softmax(torch.tanh(h @ self.W), dim=1)  # (B, T, 1)
        return h * beta

    def attention_weights(self, h: torch.Tensor) -> torch.Tensor:
        return torch.softmax(torch.tanh(h @ self.W), dim=1).squeeze(-1)


class CNNBiLSTMAttention(nn.Module):
    """Multi-task model predicting (CO2 emissions, gross generation)."""

    def __init__(
        self,
        n_bands: int = 16,
        n_steps: int = 24,
        n_static: int = 11,
        n_filters: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        fc_hidden: int = 1536,
        single_task: bool = False,
    ) -> None:
        super().__init__()
        self.single_task = single_task
        self.feature_attention = FeatureAttention(n_bands, n_steps)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(n_bands, n_filters, k, padding=pad)
                for k, pad in ((3, 1), (5, 2), (7, 3))
            ]
        )
        self.reduce = nn.Conv1d(3 * n_filters, n_filters, kernel_size=1)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(
            input_size=n_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_attention = TemporalAttention(2 * lstm_hidden)
        in_fc = n_steps * 2 * lstm_hidden + n_static  # 3072 + 11 = 3083
        n_heads = 1 if single_task else 2
        self.fc1 = nn.ModuleList([nn.Linear(in_fc, fc_hidden) for _ in range(n_heads)])
        self.fc2 = nn.ModuleList([nn.Linear(fc_hidden, 1) for _ in range(n_heads)])

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        """x_dyn: (B, n_bands, n_steps) z-scored; x_stat: (B, n_static) z-scored.

        Returns (B, 2) = (co2_metric_tons, gross_load_mwh), z-scored.
        """
        x = self.feature_attention(x_dyn)
        x = torch.cat([self.relu(conv(x)) for conv in self.convs], dim=1)
        x = self.relu(self.reduce(x))  # (B, 32, T)
        h, _ = self.bilstm(x.transpose(1, 2))  # (B, T, 128)
        h = self.temporal_attention(h)
        z = torch.cat([h.flatten(1), x_stat], dim=1)  # (B, 3083)
        outs = [fc2(self.relu(fc1(z))) for fc1, fc2 in zip(self.fc1, self.fc2)]
        return torch.cat(outs, dim=1)
