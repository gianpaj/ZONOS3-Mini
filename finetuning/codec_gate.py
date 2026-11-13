"""python -m finetuning.codec_gate --cache CACHE_DIR --out GATE_DIR

Codec reconstruction gate: decodes every cached example's codes with the matched decoder and writes the clip the
encoder saw (``<id>.input.wav``, mono 44.1 kHz) next to its reconstruction (``<id>.recon.wav``), ``report.json`` and a
listening checklist ``CHECKLIST.md``.

The numbers are a screen, not a verdict: ``mel_l1`` is the mean absolute log-mel difference (80 bands, natural log) and
``level_db`` the reconstruction's RMS level relative to the input. An LM adapter cannot restore what the codec drops,
so an event class the codec does not carry is listened to and excluded before training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from finetuning import cache as C
from finetuning.preprocess import prepare


def mel_l1(a: torch.Tensor, b: torch.Tensor, rate: int) -> float:
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(rate, n_fft=2048, hop_length=512, n_mels=80)
    return float((torch.log(mel(a) + 1e-5) - torch.log(mel(b) + 1e-5)).abs().mean())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--decoder", default=None, help="decoder id of the codec registry (default: its default)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    from zonos3.tokenizer import matched_codec
    from zonos3.tts.audio import write_pcm16_wav

    meta = C.load_meta(args.cache)
    codec, rate = meta["codec"]["name"], meta["codec"]["sample_rate"]
    entry = matched_codec.entry(codec)
    model = matched_codec.load_model(codec, torch.device(args.device), decoder=args.decoder)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for row in C.load_index(args.cache):
        with np.load(Path(args.cache) / f"{row['id']}.npz") as data:
            codes = torch.from_numpy(data["codes"].astype(np.int64))
        source = prepare(row["audio"], 600.0)[0][0]
        recon = matched_codec.decode_model(model, codes[None], entry)[0][: len(source)]
        write_pcm16_wav(out / f"{row['id']}.input.wav", source.numpy(), rate)
        write_pcm16_wav(out / f"{row['id']}.recon.wav", recon.numpy(), rate)
        level = 20 * torch.log10(recon.square().mean().sqrt() / source.square().mean().sqrt().clamp_min(1e-9))
        results.append(
            {
                "id": row["id"],
                "event": row["event"],
                "seconds": row["clip"]["seconds"],
                "source_sample_rate": row["clip"]["source_sample_rate"],
                "mel_l1": round(mel_l1(source, recon, rate), 4),
                "level_db": round(float(level), 2),
            }
        )
    by_event = defaultdict(list)
    for result in results:
        by_event[result["event"]].append(result["mel_l1"])
    summary = {event: {"clips": len(v), "mel_l1_mean": round(float(np.mean(v)), 4)} for event, v in by_event.items()}
    report = {"codec": codec, "decoder": matched_codec.decoder_id(codec, args.decoder), "summary": summary}
    (out / "report.json").write_text(json.dumps({**report, "clips": results}, indent=1) + "\n")
    lines = [
        "# Codec reconstruction listening checklist",
        "",
        "Per pair: is the event still recognizable, is its onset/offset intact, any added noise or pitch artifacts?",
        "",
        "| id | event | s | mel_l1 | event kept? | notes |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['id']} | {r['event']} | {r['seconds']} | {r['mel_l1']} | | |" for r in results]
    (out / "CHECKLIST.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
