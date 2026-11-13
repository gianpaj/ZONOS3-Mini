"""The training forward against the inference oracle, and its gradients, on a tiny random LM."""

from __future__ import annotations

import pytest
import torch
from zonos3.models.accent_format import load_adapter
from zonos3.reference.accents import AccentState
from zonos3.reference.lm import LMCache

from finetuning.lora import DEFAULT_TARGETS, LoraAdapter, target_shapes
from finetuning.model import audio_loss, forward_train


def oracle(lm, example, accents=None):
    return lm(
        example.ids,
        LMCache(),
        speaker_rows=example.speaker_rows,
        condition_rows=example.condition_rows,
        accents=accents,
    ).logits


def train_logits(lm, example, **kwargs):
    return forward_train(
        lm, example.ids, speaker_rows=example.speaker_rows, condition_rows=example.condition_rows, **kwargs
    )


def trained_adapter(lm, seed=0) -> LoraAdapter:
    """An adapter with nonzero ``up`` factors, as after training."""
    adapter = LoraAdapter(target_shapes(lm), rank=4, alpha=8.0, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for up in adapter.up:
            up.copy_(0.05 * torch.randn(up.shape, generator=generator))
    return adapter


def test_base_logits_match_the_oracle(lm, example):
    torch.testing.assert_close(train_logits(lm, example), oracle(lm, example), rtol=1e-5, atol=1e-5)


def test_fresh_adapter_leaves_logits_unchanged(lm, example):
    adapter = LoraAdapter(target_shapes(lm), rank=4, alpha=8.0)
    assert torch.equal(train_logits(lm, example, adapter=adapter), train_logits(lm, example))


def test_targets_cover_packed_and_plain_weights(lm):
    shapes = target_shapes(lm)
    assert shapes["layers.1.attention.wkv"] == (2, 32, 64)  # packed [2, kv, in]: one flat [64, 64] product
    assert {name.split(".", 2)[2] for name in shapes} == set(DEFAULT_TARGETS)
    with pytest.raises(ValueError):
        target_shapes(lm, ("feed_forward.experts",))


def test_saved_adapter_serves_through_the_oracle(lm, example, tmp_path):
    adapter = trained_adapter(lm)
    trained = train_logits(lm, example, adapter=adapter)
    assert not torch.allclose(trained, train_logits(lm, example), atol=1e-3)
    adapter.save(tmp_path / "pilot.pt")
    served = AccentState.for_model(lm, load_adapter(tmp_path / "pilot.pt"), 1.0)
    torch.testing.assert_close(oracle(lm, example, served), trained, rtol=1e-4, atol=1e-4)
    reloaded = LoraAdapter.load(tmp_path / "pilot.pt")
    torch.testing.assert_close(train_logits(lm, example, adapter=reloaded), trained, rtol=1e-5, atol=1e-5)


def test_gradients_reach_the_adapter_only(lm, example):
    adapter = LoraAdapter(target_shapes(lm), rank=4, alpha=8.0)
    loss = audio_loss(train_logits(lm, example, adapter=adapter), example.targets)
    assert torch.isfinite(loss)
    loss.backward()
    # up starts at zero: the first step moves up, and down's gradient is exactly zero
    for down, up in zip(adapter.down, adapter.up):
        assert torch.isfinite(up.grad).all() and up.grad.abs().sum() > 0
        assert down.grad is None or not down.grad.any()
    assert all(not p.requires_grad and p.grad is None for p in lm.parameters())
    adapter = trained_adapter(lm)
    audio_loss(train_logits(lm, example, adapter=adapter), example.targets).backward()
    for down, up in zip(adapter.down, adapter.up):
        assert torch.isfinite(down.grad).all() and down.grad.abs().sum() > 0 and up.grad.abs().sum() > 0


def test_layer_checkpointing_keeps_loss_and_gradients(lm, example):
    grads = []
    for checkpoint_layers in (False, True):
        adapter = trained_adapter(lm)
        loss = audio_loss(
            train_logits(lm, example, adapter=adapter, checkpoint_layers=checkpoint_layers), example.targets
        )
        loss.backward()
        grads.append((loss.detach(), [p.grad.clone() for p in adapter.parameters()]))
    torch.testing.assert_close(grads[0][0], grads[1][0])
    for a, b in zip(grads[0][1], grads[1][1]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


def test_backward_matches_finite_differences(example):
    """The backward seam of GDN, attention, AttnRes and expert dispatch: directional derivatives of the loss along
    random adapter directions against central differences. The model keeps its float32 islands (GDN state, routing
    weights, AttnRes scores), which bounds the agreement; the step is small enough that no token changes its routed
    experts (the loss is piecewise smooth in the weights)."""
    from finetuning.tests.conftest import tiny_model

    lm = tiny_model().double()
    adapter = trained_adapter(lm).double()
    params = list(adapter.parameters())

    def loss():
        logits = train_logits(lm, example, adapter=adapter)
        return torch.nn.functional.cross_entropy(logits.flatten(0, 1), example.targets.flatten(), ignore_index=-100)

    analytic = torch.autograd.grad(loss(), params)
    generator = torch.Generator().manual_seed(7)

    def shifted(directions, step):
        with torch.no_grad():
            for p, d in zip(params, directions):
                p.add_(step * d)
            value = loss()
            for p, d in zip(params, directions):
                p.sub_(step * d)
        return value

    eps = 1e-4
    for _ in range(3):
        directions = [torch.randn(p.shape, generator=generator, dtype=torch.float64) for p in params]
        expected = sum((g * d).sum() for g, d in zip(analytic, directions))
        numeric = (shifted(directions, eps) - shifted(directions, -eps)) / (2 * eps)
        torch.testing.assert_close(numeric, expected, rtol=2e-2, atol=1e-6)


def test_an_optimizer_step_moves_the_adapter_and_not_the_base(lm, example):
    adapter = trained_adapter(lm)
    before_base = {name: p.clone() for name, p in lm.named_parameters()}
    before = [p.detach().clone() for p in adapter.parameters()]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-2)
    first = audio_loss(train_logits(lm, example, adapter=adapter), example.targets)
    first.backward()
    optimizer.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, adapter.parameters()))
    assert all(torch.equal(before_base[name], p) for name, p in lm.named_parameters())
    assert audio_loss(train_logits(lm, example, adapter=adapter), example.targets) < first
