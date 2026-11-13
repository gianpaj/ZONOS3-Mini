"""python -m finetuning.evaluate --cache CACHE_DIR --split heldout [--adapter OUT_DIR/pilot.pt]

Teacher-forced audio loss (nats per supervised code, finetuning.model.audio_loss) of the base LM and, with
``--adapter``, of the adapter, per event label of a split. One JSON line per event and one for the split.

A lower held-out loss means the adapter predicts unseen clips of that event better under teacher forcing. It does not
show that sampled audio contains the event; that needs listening or an event classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import torch

from finetuning import cache as C
from finetuning.train import load_lm, sequence_loss


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--speaker-mode", choices=C.SPEAKER_MODES, default="embedding")
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    lm, tokenizer_sha = load_lm(args.model, args.device, args.dtype)
    C.check_base(C.load_meta(args.cache), lm.config, tokenizer_sha)
    adapter = None
    if args.adapter:
        from finetuning.lora import LoraAdapter

        adapter = LoraAdapter.load(args.adapter).to(args.device)
    losses = defaultdict(lambda: {"base": [], "adapter": []})
    for row in C.load_index(args.cache, args.split):
        if row["frames"] > args.max_frames:
            continue
        seq = C.load_sequence(args.cache, row, lm.config, args.speaker_mode)
        with torch.no_grad():
            losses[row["event"]]["base"].append(sequence_loss(lm, seq).item())
            if adapter is not None:
                losses[row["event"]]["adapter"].append(sequence_loss(lm, seq, adapter).item())

    def line(event, groups):
        base = [v for g in groups for v in g["base"]]
        tuned = [v for g in groups for v in g["adapter"]]
        doc = {"split": args.split, "event": event, "clips": len(base), "base_loss": round(sum(base) / len(base), 4)}
        if tuned:
            doc["adapter_loss"] = round(sum(tuned) / len(tuned), 4)
            doc["clips_improved"] = sum(t < b for t, b in zip(tuned, base))
        return json.dumps(doc)

    for event in sorted(losses):
        print(line(event, [losses[event]]))
    if losses:
        print(line("all", list(losses.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
