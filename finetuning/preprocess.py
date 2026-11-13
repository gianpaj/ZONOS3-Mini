"""python -m finetuning.preprocess --manifest data.jsonl --cache CACHE_DIR

Turns a manifest (finetuning.manifest) into the token cache (finetuning.cache) with the LM's frozen front ends:

- text: the request normalizer of the row's language, then the checkpoint's SuperBPE ids;
- audio: mono, 44.1 kHz (torchaudio resampling on the CPU), peak-normalized only above 1.0, then the matched dual
  codec encoder with its semantic teacher (zonos3.tokenizer.dual_reference, CUDA only);
- reference clip: the MECT speaker embedding (CPU) and the clip's codec codes, cut to ``--ref-seconds``.

The codec encoder needs several GB of GPU memory and runs in this process only: the trainer reads the cache.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from finetuning.manifest import Row, load_manifest

VERSION = 1
_TAG = re.compile(r"\[[^\[\]\s]+\]")


def prepare(path: str, max_seconds: float) -> tuple[torch.Tensor, dict]:
    """The clip as the codec encoder gets it (mono [1, samples] at 44.1 kHz) and what was done to it."""
    import soundfile as sf
    from zonos3.tokenizer.dual_reference import CODEC_RATE, prepare_reference

    wav, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(wav.T.copy())
    peak = float(wave.mean(0).abs().amax()) if wave.numel() else 0.0
    prepared = prepare_reference(wave, int(rate), max_seconds)
    info = {
        "source_sample_rate": int(rate),
        "source_channels": int(wave.shape[0]),
        "source_seconds": round(wave.shape[1] / rate, 4),
        "resampled": int(rate) != CODEC_RATE,
        "downmixed": wave.shape[0] > 1,
        "peak_normalized": peak > 1.0,
        "seconds": round(prepared.shape[1] / CODEC_RATE, 4),
    }
    return prepared, info


def normalize_text(row: Row, normalizer) -> str:
    """The row's text as a request would read it; an event tag ``[name]`` must come through unchanged."""
    from zonos3.tokenizer.textnorm import normalize_request_text

    text = normalize_request_text(row.text, row.language, normalizer)
    lost = [tag for tag in _TAG.findall(row.text) if tag not in text]
    if lost:
        raise ValueError(f"{row.id}: text normalization changed the event tags {lost}: {text!r}")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", default=None, help="LM checkpoint directory or hf:// folder (default: the release)")
    parser.add_argument("--device", default="cuda:0", help="codec encoder device (CUDA)")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="longest target clip; longer rows are refused")
    parser.add_argument("--ref-seconds", type=float, default=10.0, help="reference clips are cut to this length")
    parser.add_argument("--min-free-gb", type=float, default=7.0, help="refuse to load the encoder with less free VRAM")
    args = parser.parse_args(argv)

    from zonos3.models.mect_speaker import registry_entry
    from zonos3.models.speaker_cloning import create_speaker_embedder
    from zonos3.reference.lm import Zonos3LMConfig
    from zonos3.tokenizer import dual_reference, matched_codec
    from zonos3.tokenizer.textnorm import load_normalizer
    from zonos3.tts.superbpe import load_spec, text_ids
    from zonos3.utils import release
    from zonos3.utils.hashing import sha256_file

    rows = load_manifest(args.manifest)
    model = args.model or release.RELEASE_LM
    root = Path(release.resolve_dir_or_local(model))
    config = Zonos3LMConfig.from_params(json.loads((root / "params.json").read_text()))
    spec = load_spec(root)
    normalizer = load_normalizer()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    # Text and audio preparation first: a bad row fails before the encoder loads.
    prepared = {}
    for row in rows:
        text = normalize_text(row, normalizer)
        wave, info = prepare(row.audio, 600.0)
        if info["seconds"] > args.max_seconds:
            raise ValueError(f"{row.id}: {info['seconds']} s exceeds --max-seconds {args.max_seconds}")
        prepared[row.id] = (text, text_ids(text, spec), wave, info)

    free = torch.cuda.mem_get_info(torch.device(args.device))[0] / 2**30
    if free < args.min_free_gb:
        print(f"{free:.1f} GiB free on {args.device}, {args.min_free_gb} needed for the codec encoder", file=sys.stderr)
        return 2
    entry = matched_codec.entry(config.audio_codec)
    generator = release.resolve_file(entry["checkpoint"])
    actual = sha256_file(generator)
    if actual != entry["checkpoint_sha256"]:
        raise ValueError(f"codec generator sha256 {actual} != registry {entry['checkpoint_sha256']}")
    speaker = create_speaker_embedder(config.speaker_embedding_codec, config.speaker_embedding_dim, device="cpu")
    references: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}

    def reference(path: str):
        """Speaker embedding of the clip as a server computes it (the file's own rate, first --ref-seconds) and the
        clip's codec codes."""
        if path not in references:
            import soundfile as sf

            wav, rate = sf.read(
                path, dtype="float32", always_2d=True, frames=int(args.ref_seconds * sf.info(path).samplerate)
            )
            with torch.inference_mode():
                embedding = speaker(torch.from_numpy(wav.T.copy()), int(rate)).reshape(-1).numpy().astype(np.float32)
            wave, info = prepare(path, args.ref_seconds)
            codes = dual_reference.encode_dual_reference(
                wave, dual_reference.CODEC_RATE, max_seconds=args.ref_seconds, device=args.device
            )
            references[path] = (embedding, codes.numpy().astype(np.int16), info)
        return references[path]

    index = []
    for row in rows:
        text, ids, wave, info = prepared[row.id]
        codes = dual_reference.encode_dual_reference(
            wave, dual_reference.CODEC_RATE, max_seconds=600.0, device=args.device
        )
        arrays = {"codes": codes.numpy().astype(np.int16), "text_ids": np.asarray(ids, dtype=np.int32)}
        record = {
            **row.to_json(),
            "normalized_text": text,
            "text_tokens": len(ids),
            "frames": len(codes),
            "clip": {**info, "sha256": sha256_file(row.audio)},
        }
        if row.ref_audio:
            embedding, ref_codes, ref_info = reference(row.ref_audio)
            arrays.update(speaker_embedding=embedding, ref_codes=ref_codes)
            record["reference"] = {**ref_info, "frames": len(ref_codes), "sha256": sha256_file(row.ref_audio)}
        np.savez(cache / f"{row.id}.npz", **arrays)
        index.append(record)
        print(f"{row.id}: {info['seconds']} s -> {len(codes)} frames, {len(ids)} text ids", flush=True)

    teacher = Path(release.resolve_dir_or_local(entry["teacher"]))
    meta = {
        "version": VERSION,
        "model": model,
        "codec": {
            "name": config.audio_codec,
            "generator_sha256": actual,
            "teacher_sha256": sha256_file(teacher / "model.safetensors"),
            "codebooks": config.n_codebooks,
            "codebook_size": config.codebook_size,
            "hop": dual_reference.HOP,
            "sample_rate": dual_reference.CODEC_RATE,
            "encoder": dual_reference.get_encoder(args.device).metadata()["precision"],
        },
        "tokenizer": {"sha256": spec["sha256"], "normalization": normalizer is not None},
        "speaker_encoder": {
            "codec": config.speaker_embedding_codec,
            "sha256": registry_entry(config.speaker_embedding_codec)["sha256"],
        },
        "max_seconds": args.max_seconds,
        "ref_seconds": args.ref_seconds,
        "torch": torch.__version__,
        "peak_cuda_gib": round(torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**30, 2),
    }
    (cache / "index.jsonl").write_text("".join(json.dumps(record) + "\n" for record in index))
    (cache / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(index)} examples -> {cache} (encoder peak {meta['peak_cuda_gib']} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
