"""python -m finetuning.train --cache CACHE_DIR --out OUT_DIR --name pilot [--steps N]

LoRA fine-tuning of the ZONOS3 Mini LM on a token cache (finetuning.preprocess), one sequence per forward.

The default is a one-step smoke run. ``--steps`` up to 50 is the capped overfit diagnostic; more needs ``--long``.
Writes ``OUT_DIR/<name>.pt`` (a ``zonos-accent-lora/1`` adapter: serve it as accent ``<name>``),
``OUT_DIR/<name>.json`` (settings, base and cache identity, trainable tensors, losses, memory) and
``OUT_DIR/<name>.check/logits.pt`` (logits of one sequence before saving, for ``finetuning.check``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from finetuning import cache as C
from finetuning.lora import DEFAULT_TARGETS, LoraAdapter, target_shapes
from finetuning.model import audio_loss, forward_train

SMOKE_CAP = 50


def load_lm(model: str | None, device: str, dtype: str):
    """The base LM (strict load, every parameter frozen) and its tokenizer hash."""
    from zonos3.reference.lm import Zonos3LM
    from zonos3.utils import release

    lm = Zonos3LM.load(
        model or release.RELEASE_LM, device=device, dtype={"bf16": torch.bfloat16, "f32": torch.float32}[dtype]
    )
    assert not any(p.requires_grad for p in lm.parameters())
    return lm, (lm.config.text_tokenizer_spec or {}).get("sha256")


def sequence_loss(lm, seq, adapter=None, checkpoint_layers: bool = False) -> torch.Tensor:
    logits = forward_train(
        lm,
        seq.ids,
        speaker_rows=seq.speaker_rows,
        condition_rows=seq.condition_rows,
        adapter=adapter,
        checkpoint_layers=checkpoint_layers,
    )
    return audio_loss(logits, seq.targets)


def gpu_processes() -> list[dict]:
    """Compute processes on the GPUs with their memory (MiB), this one included."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = [[field.strip() for field in line.split(",")] for line in out.splitlines() if line.strip()]
    return [
        {"pid": int(pid), "name": Path(name).name, "mib": int(mib), "self": int(pid) == os.getpid()}
        for pid, name, mib in rows
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="pilot", help="adapter file stem, the accent name at inference")
    parser.add_argument("--model", default=None, help="LM checkpoint directory or hf:// folder (default: the release)")
    parser.add_argument("--split", default="train", help="index split to train on (train or overfit)")
    parser.add_argument("--steps", type=int, default=1, help="optimizer steps (default: a one-step smoke run)")
    parser.add_argument("--long", action="store_true", help=f"allow more than {SMOKE_CAP} steps")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=1, help="sequences per optimizer step")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=150, help="skip examples with more audio frames (15 per s)")
    parser.add_argument("--speaker-mode", choices=C.SPEAKER_MODES, default="embedding")
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16", help="frozen base weights and activations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-layers", action="store_true", help="recompute each layer in the backward pass")
    parser.add_argument("--save-every", type=int, default=0, help="also save every N steps (0 = at the end)")
    args = parser.parse_args(argv)
    if args.steps > SMOKE_CAP and not args.long:
        parser.error(f"--steps {args.steps} exceeds the {SMOKE_CAP}-step diagnostic cap; pass --long for a real run")

    cuda = torch.device(args.device).type == "cuda"
    torch.manual_seed(args.seed)
    lm, tokenizer_sha = load_lm(args.model, args.device, args.dtype)
    config = lm.config
    meta = C.load_meta(args.cache)
    C.check_base(meta, config, tokenizer_sha)
    rows = C.load_index(args.cache, args.split)
    skipped = [row["id"] for row in rows if row["frames"] > args.max_frames]
    rows = [row for row in rows if row["frames"] <= args.max_frames]
    if not rows:
        parser.error(f"no {args.split!r} example of at most {args.max_frames} frames in {args.cache}")
    sequences = [C.load_sequence(args.cache, row, config, args.speaker_mode) for row in rows]

    adapter = LoraAdapter(target_shapes(lm, tuple(args.targets.split(","))), args.rank, args.alpha, args.seed)
    adapter.to(args.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainable = sum(p.numel() for p in adapter.parameters())
    weights_bytes = torch.cuda.memory_allocated() if cuda else 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    order: list[int] = []
    losses, first_grads = [], None
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.grad_accum):
            if not order:
                order = torch.randperm(len(sequences), generator=generator).tolist()
            loss = sequence_loss(lm, sequences[order.pop()], adapter, args.checkpoint_layers)
            if not torch.isfinite(loss):
                raise RuntimeError(f"step {step}: loss is {loss.item()}")
            (loss / args.grad_accum).backward()
            total += loss.item() / args.grad_accum
        if first_grads is None:
            first_grads = {
                "up_nonzero": sum(bool(p.grad.any()) for p in adapter.up),
                "down_nonzero": sum(p.grad is not None and bool(p.grad.any()) for p in adapter.down),
                "finite": all(bool(torch.isfinite(p.grad).all()) for p in adapter.parameters() if p.grad is not None),
            }
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(round(total, 5))
        print(f"step {step}/{args.steps} loss {total:.4f} grad_norm {norm.item():.3f}", flush=True)
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            adapter.save(out / f"{args.name}.pt")
    seconds = time.monotonic() - started
    assert not any(p.requires_grad or p.grad is not None for p in lm.parameters())

    memory = None
    if cuda:
        peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        longest = max(sequences, key=lambda s: len(s.ids))
        logits_bytes = (
            len(longest.ids) * config.n_codebooks * config.audio_vocab * ((2 if args.dtype == "bf16" else 4) + 4)
        )
        state = 4 * trainable  # float32 factors; AdamW keeps two moments, the backward one gradient
        memory = {
            "measured_mib": {
                "frozen_weights_and_adapter_after_load": round(weights_bytes / 2**20, 1),
                "peak_allocated": round(peak / 2**20, 1),
                "peak_reserved": round(reserved / 2**20, 1),
                "allocator_overhead_at_peak": round((reserved - peak) / 2**20, 1),
            },
            "estimated_mib": {
                "trainable_weights": round(state / 2**20, 1),
                "gradients": round(state / 2**20, 1),
                "optimizer_state": round(2 * state / 2**20, 1),
                "logits_longest_sequence": round(logits_bytes / 2**20, 1),
                "activations_and_workspaces": round((peak - weights_bytes - 3 * state - logits_bytes) / 2**20, 1),
            },
            "longest_sequence_rows": len(longest.ids),
            "processes": gpu_processes(),
            "device": torch.cuda.get_device_name(),
        }

    check = sequences[0]
    with torch.no_grad():
        logits = forward_train(
            lm, check.ids, speaker_rows=check.speaker_rows, condition_rows=check.condition_rows, adapter=adapter
        )
    adapter.save(out / f"{args.name}.pt")
    (out / f"{args.name}.check").mkdir(exist_ok=True)
    torch.save({"id": rows[0]["id"], "logits": logits.float().cpu()}, out / f"{args.name}.check" / "logits.pt")
    report = {
        "args": vars(args),
        "base": {
            "model": str(lm.root),
            "tokenizer_sha256": tokenizer_sha,
            "parameters": sum(p.numel() for p in lm.parameters()),
        },
        "cache": meta,
        "examples": {"used": [row["id"] for row in rows], "skipped_over_max_frames": skipped},
        "trainable": {"parameters": trainable, "tensors": {m: list(s) for m, s in adapter.shapes.items()}},
        "first_step_gradients": first_grads,
        "losses": losses,
        "seconds_per_step": round(seconds / args.steps, 3),
        "memory": memory,
        "torch": torch.__version__,
    }
    (out / f"{args.name}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        f"{trainable:,} trainable parameters, loss {losses[0]} -> {losses[-1]}, adapter -> {out / (args.name + '.pt')}"
    )
    if memory:
        print(json.dumps(memory["measured_mib"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
