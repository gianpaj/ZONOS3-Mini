"""Tiny random LM shared by the fine-tuning tests (CPU, float32)."""

from __future__ import annotations

import pytest
import torch
from zonos3.reference.lm import Zonos3LM, Zonos3LMConfig

N_CB, CODEBOOK, TEXT_VOCAB = 10, 16, 460

TINY = {
    "n_layers": 4,
    "dim": 64,
    "head_dim": 16,
    "n_heads": 4,
    "n_kv_heads": 2,
    "norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_seqlen": 256,
    "loss_softcap": 15.0,
    "ffn_dim_multiplier": 2.0,
    "multiple_of": 32,
    "moe_ffn_dim_multiplier": 1.0,
    "moe_multiple_of": 32,
    "moe_alpha": 0.5,
    "moe_n_experts": 4,
    "n_codebooks": N_CB,
    "codebook_size": CODEBOOK,
    "codebook_sizes": [CODEBOOK] * N_CB,
    "eoa_id": CODEBOOK,
    "audio_pad_id": CODEBOOK + 1,
    "text_vocab": TEXT_VOCAB,
    "speaker_embedding_dim": 8,
    "attention_pattern": "gdn1_global1",  # layers 0 and 2 GDN, layers 1 and 3 global attention
    "residual_type": "full_attnres",
    "use_xsa": False,
    "speaker_enabled": True,
    "condition_tokens_enabled": True,
    "moe_router_type": "linear",
    "moe_router_use_eda": False,
    "moe_balancing_strategy": "quantile",
    "moe_router_topk": 2,
    "moe_start_from_layer": 1,
    "moe_end_from_layer": 0,
    "condition_count": 3,
    "condition_n_buckets": 4,
    "condition_names": ["loudness", "fmax", "speaking_rate"],
}


def tiny_model(seed: int = 0) -> Zonos3LM:
    model = Zonos3LM(Zonos3LMConfig.from_params({"model": TINY}))
    generator = torch.Generator().manual_seed(seed)
    for name, param in model.named_parameters():
        noise = torch.randn(param.shape, generator=generator)
        if name.endswith(("norm.weight", "rmsnorm_eda.weight", "out_weight")):
            param.data = 1.0 + 0.1 * noise
        elif name.endswith("balancing_biases"):
            param.data = 0.1 * noise
        else:
            param.data = 0.2 * noise
    return model.eval()


@pytest.fixture
def lm() -> Zonos3LM:
    return tiny_model()


@pytest.fixture
def example(lm):
    """A sequence with a speaker row, condition rows, 5 text tokens and 7 audio frames."""
    from finetuning.sequence import build_sequence

    g = torch.Generator().manual_seed(1)
    codes = torch.randint(0, CODEBOOK, (7, N_CB), generator=g)
    text_ids = [2, *torch.randint(4, TEXT_VOCAB, (5,), generator=g).tolist(), 3]
    return build_sequence(lm.config, text_ids, codes, speaker_embedding=torch.linspace(-1, 1, 8))
