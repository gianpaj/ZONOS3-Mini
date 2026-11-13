"""python -m finetuning.request --text "Hello [laugh] there." [--accent pilot] [--cache CACHE_DIR --example ID] -o req.json

Writes a v2 GenerateRequest for the reference synthesizer, which serves a trained adapter as an accent::

    python -m zonos3.reference --model hf://Zyphra/ZONOS3-Mini/lm/zonos3_mini --phoneme-rows '' \\
        --accent-adapters OUT_DIR --request req.json -o out.wav

``--example`` takes the speaker embedding of a cached example (its reference clip), the conditioning the trainer used
with ``--speaker-mode embedding``. Without ``--accent`` the request runs the base model.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="en_us")
    parser.add_argument("--accent", default=None, help="adapter name (the .pt file stem)")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=None, help="frame budget (15 frames per second)")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--example", default=None, help="cached example whose speaker embedding conditions the request")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args(argv)

    request: dict = {"text": args.text, "language": args.language, "seed": args.seed}
    if args.max_tokens:
        request["max_tokens"] = args.max_tokens
    if args.accent:
        request.update(accent=args.accent, accent_strength=args.strength)
    if args.example:
        if not args.cache:
            parser.error("--example needs --cache")
        with np.load(Path(args.cache) / f"{args.example}.npz") as data:
            if "speaker_embedding" not in data:
                parser.error(f"{args.example} has no reference clip")
            buffer = io.BytesIO()
            np.save(buffer, data["speaker_embedding"])
        request.update(
            speaker_mode="embedding",
            speaker_embedding_base64=base64.b64encode(buffer.getvalue()).decode(),
            speaker_embedding_name=f"{args.example}.npy",
        )
    Path(args.out).write_text(json.dumps(request) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
