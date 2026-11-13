"""Teacher-forced training forward of the ZONOS3 Mini LM.

``zonos3.reference.lm.Zonos3LM.forward`` is the inference oracle: it runs under ``torch.no_grad``, continues a cache
and dispatches MoE experts one (token, expert) pair at a time. ``forward_train`` computes the same logits for one whole
sequence that starts at position 0, with autograd on:

- no cache: the GDN state and conv history start at zero, attention sees the sequence's own rows only;
- experts run once per routed expert over the tokens that chose it;
- every adapted linear goes through ``adapter.linear(module, x, y)``, the hook ``zonos3.reference.accents.AccentState``
  serves, under the checkpoint's module names;
- ``checkpoint_layers`` recomputes each layer in the backward pass instead of keeping its activations.

One call is one sequence, so no state or attention crosses examples. The router's expert choice is a hard top-k; its
gradient reaches earlier layers through the softmax over the chosen logits only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from zonos3.reference.lm import GDN_HEAD_DIM, GDN_HEADS, GDN_KERNEL, Zonos3LM, full_attnres_route, rope_interleaved

IGNORE = -100


def _linear(adapter, module: str, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    y = F.linear(x, weight)
    return y if adapter is None else adapter.linear(module, x, y)


def _gdn(lm: Zonos3LM, adapter, x: torch.Tensor, layer: int) -> torch.Tensor:
    a, p = lm.layers[layer].attention, f"layers.{layer}.attention."
    heads, hd, kernel = GDN_HEADS, GDN_HEAD_DIM, GDN_KERNEL
    inner, n = heads * hd, len(x)
    projection = _linear(adapter, p + "in_proj", x, a.in_proj.weight)
    qkv, z, aa, b = projection.split((3 * inner, inner, heads, heads), -1)
    history = torch.zeros(3 * inner, kernel - 1, dtype=x.dtype, device=x.device)
    conv = F.conv1d(torch.cat((history, qkv.T), -1).unsqueeze(0), a.conv.weight, groups=3 * inner)
    conv = F.silu(conv.squeeze(0).T)
    q, k, v = [part.reshape(n, heads, hd).float() for part in conv.split(inner, -1)]
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    decay = torch.exp(-a.A_log.float().exp() * F.softplus(aa.float() + a.dt_bias.float()))
    beta = b.float().sigmoid()
    state = torch.zeros(heads, hd, hd, dtype=torch.float32, device=x.device)
    outputs = []
    for token in range(n):  # [heads, value, key], float32, as the oracle
        state = state * decay[token, :, None, None]
        prediction = (state * k[token, :, None, :]).sum(-1)
        delta = beta[token, :, None] * (v[token] - prediction)
        state = state + delta[:, :, None] * k[token, :, None, :]
        outputs.append((state * (q[token] * hd**-0.5)[:, None, :]).sum(-1))
    recurrent = torch.stack(outputs).to(x.dtype)
    normalized = lm._rms(recurrent, a.out_weight, eps=1e-5)
    gated = (normalized.float() * F.silu(z.reshape(n, heads, hd).float())).to(x.dtype).reshape(n, inner)
    return _linear(adapter, p + "wo", gated, a.wo.weight)


def _attention(lm: Zonos3LM, adapter, x: torch.Tensor, layer: int) -> torch.Tensor:
    c, a, p = lm.config, lm.layers[layer].attention, f"layers.{layer}.attention."
    n, nh, nkv, hd = len(x), c.n_heads, c.n_kv_heads, c.head_dim
    gate = _linear(adapter, p + "gater", x, a.gater.weight).sigmoid()
    q = _linear(adapter, p + "wq", x, a.wq.weight).view(n, nh, hd)
    kv = _linear(adapter, p + "wkv", x, a.wkv.weight.reshape(2 * nkv * hd, c.dim))
    k, v = kv[:, : nkv * hd].reshape(n, nkv, hd), kv[:, nkv * hd :].reshape(n, nkv, hd)
    q = lm._rms(q, eps=1e-6) * a.temp.abs().view(1, nh, 1)
    k = lm._rms(k, eps=1e-6)
    positions = torch.arange(n, device=x.device)
    q = rope_interleaved(q, positions, c.rope_theta)
    k = rope_interleaved(k, positions, c.rope_theta)
    rep = nh // nkv
    key = k.to(q.dtype).repeat_interleave(rep, 1).transpose(0, 1)
    value = v.to(q.dtype).repeat_interleave(rep, 1).transpose(0, 1)
    causal = positions[None, :] <= positions[:, None]
    out = F.scaled_dot_product_attention(q.transpose(0, 1), key, value, attn_mask=causal, scale=hd**-0.5)
    return _linear(adapter, p + "wo", (out.transpose(0, 1) * gate.unsqueeze(-1)).reshape(n, nh * hd), a.wo.weight)


def _feed_forward(lm: Zonos3LM, adapter, x: torch.Tensor, layer: int) -> torch.Tensor:
    c, f, p = lm.config, lm.layers[layer].feed_forward, f"layers.{layer}.feed_forward."
    if layer == 0:
        hidden = c.ffn_hidden
        flat = _linear(adapter, p + "w_in", x, f.w_in.weight.reshape(2 * hidden, -1))
        return _linear(adapter, p + "w_out", flat[:, :hidden] * F.silu(flat[:, hidden:]), f.w_out.weight)
    logits = _linear(
        adapter, p + "router.router_linear", lm._rms(x, f.router.rmsnorm_eda.weight), f.router.router_linear.weight
    )
    logits = logits.float()
    choices = torch.topk(logits - f.router.balancing_biases.float(), c.moe_topk, -1).indices
    weights = logits.gather(-1, choices).softmax(-1).reshape(-1)
    latent = _linear(adapter, p + "latent_down", x, f.latent_down.weight)
    gate_up_proj, down_proj = f.experts.gate_up_proj, f.experts.down_proj
    half = gate_up_proj.shape[1] // 2
    expert = choices.reshape(-1)
    token = torch.arange(len(x), device=x.device).repeat_interleave(c.moe_topk)
    order = torch.argsort(expert, stable=True)
    terms, start = [], 0
    for e, count in enumerate(torch.bincount(expert, minlength=c.n_experts).tolist()):
        if not count:
            continue
        pairs = order[start : start + count]
        start += count
        gate_up = F.linear(latent[token[pairs]], gate_up_proj[e])
        out = F.linear(F.silu(gate_up[:, :half]) * gate_up[:, half:], down_proj[e])
        terms.append((out.float() * weights[pairs, None]).to(x.dtype))
    output = torch.zeros_like(latent).index_add(0, token[order], torch.cat(terms))
    return _linear(adapter, p + "latent_up", output, f.latent_up.weight)


def forward_train(
    lm: Zonos3LM,
    ids: torch.Tensor,
    *,
    speaker_rows: dict[int, torch.Tensor] | None = None,
    condition_rows: dict[int, int] | None = None,
    adapter=None,
    checkpoint_layers: bool = False,
) -> torch.Tensor:
    """Softcapped logits [rows, n_codebooks, audio_vocab] of one sequence ``ids`` [rows, n_codebooks + 1] from row 0.
    Row r's logits predict the audio columns of row r + 1. ``speaker_rows`` / ``condition_rows`` as in the oracle."""
    c = lm.config
    ids = ids.to(lm.device, torch.long)
    if len(ids) > c.max_seqlen:
        raise ValueError(f"{len(ids)} rows exceed the model context {c.max_seqlen}")
    embedders = lm.multi_embedder.embedders
    x = sum(F.embedding(ids[:, cb], embedders[cb].weight) for cb in range(c.n_codebooks + 1))
    x = lm._rms(lm._replace_rows(x, 0, speaker_rows, condition_rows))
    queries = lm.block_attnres_queries

    def block(layer: int, *sources: torch.Tensor):
        sources = list(sources)
        mixed = full_attnres_route(sources, queries[2 * layer], c.norm_eps)
        attn_in = lm._rms(mixed, lm.layers[layer].attention_norm.weight)
        attention = (_gdn if c.uses_gdn(layer) else _attention)(lm, adapter, attn_in, layer)
        mixed = full_attnres_route(sources + [attention], queries[2 * layer + 1], c.norm_eps)
        return attention, _feed_forward(lm, adapter, lm._rms(mixed, lm.layers[layer].ffn_norm.weight), layer)

    sources = [x]
    for layer in range(c.n_layers):
        if checkpoint_layers:
            sources.extend(checkpoint(block, layer, *sources, use_reentrant=False))
        else:
            sources.extend(block(layer, *sources))
    mixed = full_attnres_route(sources, queries[-1], c.norm_eps)
    logits = _linear(adapter, "multi_output", lm._rms(mixed, lm.out_norm.weight), lm.multi_output.weight)
    return c.loss_softcap * torch.tanh(logits.view(len(ids), c.n_codebooks, c.audio_vocab) / c.loss_softcap)


def audio_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over every supervised (row, codebook) position of ``targets`` [rows, n_codebooks]
    (``IGNORE`` elsewhere), float32, every codebook weighted alike."""
    return F.cross_entropy(logits.float().flatten(0, 1), targets.to(logits.device).flatten(), ignore_index=IGNORE)
