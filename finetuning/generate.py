"""python -m finetuning.generate --accent-adapters OUT_DIR --out-dir WAV_DIR req1.json [req2.json ...]

Synthesizes v2 GenerateRequests (finetuning.request) with the reference synthesizer in one process and writes
``WAV_DIR/<request stem>.wav`` (16-bit mono 44.1 kHz). The same path as ``python -m zonos3.reference``, which loads
the LM and the decoder once per request.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("requests", nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--accent-adapters", default=None, help="adapter directory (<name>.pt files)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    from zonos3.generation import SAMPLE_RATE
    from zonos3.reference.synth import ReferenceAssets, synthesize
    from zonos3.tts.audio import write_pcm16_wav

    from finetuning.train import load_lm

    lm, _ = load_lm(args.model, args.device, args.dtype)
    assets = ReferenceAssets.load(accent_adapters=args.accent_adapters)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in map(Path, args.requests):
        result = synthesize(lm, json.loads(path.read_text()), assets=assets, device=args.device)
        write_pcm16_wav(out / f"{path.stem}.wav", result.pcm, SAMPLE_RATE)
        print(
            json.dumps(
                {
                    "request": path.name,
                    "seconds": round(len(result.pcm) / SAMPLE_RATE, 2),
                    "rows": len(result.rows),
                    "eos_frame": result.eos_frame,
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
