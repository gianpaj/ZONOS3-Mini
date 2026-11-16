"""python -m finetuning.check --cache CACHE_DIR [--adapter OUT_DIR/pilot.pt] [--dtype f32|bf16]

Compares the training forward with the inference oracle (zonos3.reference.lm.Zonos3LM.forward) on cached sequences:

- base: ``forward_train`` without an adapter against the oracle;
- with ``--adapter``, in a fresh process: the adapter reloaded for training against the logits the trainer stored
  before it saved (``<adapter>.check/logits.pt``), and against the oracle serving the same file as an accent at
  strength 1 (zonos3.models.accent_format.load_adapter, zonos3.reference.accents.AccentState).

Prints one JSON line per comparison: largest and mean absolute logit difference, mean KL divergence of the per-codebook
distributions (reference || candidate, nats) and top-1 agreement, over the supervised positions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from finetuning import cache as C
from finetuning.model import IGNORE, forward_train
from finetuning.train import load_lm


def compare(reference: torch.Tensor, candidate: torch.Tensor, targets: torch.Tensor) -> dict:
    keep = (targets != IGNORE).to(reference.device)
    a, b = reference.float()[keep], candidate.float()[keep]
    log_a, log_b = a.log_softmax(-1), b.log_softmax(-1)
    return {
        "positions": int(keep.sum()),
        "max_abs": float((a - b).abs().max()),
        "mean_abs": float((a - b).abs().mean()),
        "mean_kl": float((log_a.exp() * (log_a - log_b)).sum(-1).mean()),
        "top1_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--adapter", default=None, help="an adapter the trainer saved")
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default=None, help="index split (default: all)")
    parser.add_argument("--examples", type=int, default=2, help="sequences for the base comparison")
    parser.add_argument("--speaker-mode", choices=C.SPEAKER_MODES, default="embedding")
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="f32")
    parser.add_argument("--oracle-dtype", choices=("bf16", "f32"), default=None, help="default: --dtype")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    from zonos3.reference.lm import LMCache

    lm, tokenizer_sha = load_lm(args.model, args.device, args.dtype)
    C.check_base(C.load_meta(args.cache), lm.config, tokenizer_sha)
    oracle_lm = lm
    if args.oracle_dtype not in (None, args.dtype):
        oracle_lm, _ = load_lm(args.model, args.device, args.oracle_dtype)
    rows = C.load_index(args.cache, args.split)

    def sequence(row):
        return C.load_sequence(args.cache, row, lm.config, args.speaker_mode)

    def oracle(seq, accents=None):
        return oracle_lm(
            seq.ids, LMCache(), speaker_rows=seq.speaker_rows, condition_rows=seq.condition_rows, accents=accents
        ).logits

    def train(seq, adapter=None):
        with torch.no_grad():
            return forward_train(
                lm, seq.ids, speaker_rows=seq.speaker_rows, condition_rows=seq.condition_rows, adapter=adapter
            )

    for row in rows[: args.examples]:
        seq = sequence(row)
        result = compare(oracle(seq), train(seq), seq.targets)
        print(json.dumps({"check": "base_train_vs_oracle", "id": row["id"], "rows": len(seq.ids), **result}))
        if oracle_lm is not lm:  # the oracle's own dtype gap, to read the line above against
            own = lm(seq.ids, LMCache(), speaker_rows=seq.speaker_rows, condition_rows=seq.condition_rows).logits
            print(json.dumps({"check": "oracle_dtype_gap", "id": row["id"], **compare(oracle(seq), own, seq.targets)}))

    if args.adapter:
        from zonos3.models.accent_format import load_adapter
        from zonos3.reference.accents import AccentState

        from finetuning.lora import LoraAdapter

        path = Path(args.adapter)
        stored = torch.load(path.with_suffix(".check") / "logits.pt", weights_only=True)
        row = next(row for row in rows if row["id"] == stored["id"])
        seq = sequence(row)
        reloaded = train(seq, LoraAdapter.load(path).to(args.device))
        served = oracle(seq, AccentState.for_model(oracle_lm, load_adapter(path), 1.0))
        base = train(seq)
        for name, reference, candidate in (
            ("reloaded_vs_presave", stored["logits"].to(reloaded.device), reloaded),
            ("served_accent_vs_reloaded", reloaded, served),
            ("adapter_vs_base", base, reloaded),
        ):
            print(json.dumps({"check": name, "id": row["id"], **compare(reference, candidate, seq.targets)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
