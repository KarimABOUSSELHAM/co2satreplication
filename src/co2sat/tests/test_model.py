"""Shape and sanity tests for the CNN-BiLSTM-Attention model."""

import torch

from co2sat.model import CNNBiLSTMAttention, FeatureAttention, TemporalAttention


def test_forward_shape() -> None:
    model = CNNBiLSTMAttention()
    out = model(torch.randn(4, 16, 24), torch.randn(4, 11))
    assert out.shape == (4, 2)


def test_fc_input_matches_table_s1() -> None:
    """24*128 + 11 = 3083, the dense input dimension of Supplementary Table S1."""
    model = CNNBiLSTMAttention()
    assert model.fc1[0].in_features == 3083
    assert model.fc1[0].out_features == 1536


def test_band_subset_constructor() -> None:
    """The SEVIRI ablation (12 bands) must build and run without reshaping."""
    model = CNNBiLSTMAttention(n_bands=12)
    out = model(torch.randn(2, 12, 24), torch.randn(2, 11))
    assert out.shape == (2, 2)


def test_attention_weights_normalized() -> None:
    fa = FeatureAttention()
    alpha = fa.attention_maps(torch.randn(3, 16, 24))
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(3, 24), atol=1e-5)
    ta = TemporalAttention()
    beta = ta.attention_weights(torch.randn(3, 24, 128))
    assert torch.allclose(beta.sum(dim=1), torch.ones(3), atol=1e-5)
