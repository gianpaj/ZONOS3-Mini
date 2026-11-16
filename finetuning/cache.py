"""The token cache ``finetuning.preprocess`` writes and the trainer reads.

``<cache>/meta.json``: codec, checkpoint hashes, tokenizer and preprocessing settings. ``<cache>/index.jsonl``: one row
per example (the manifest row plus what preprocessing measured). ``<cache>/<id>.npz``: ``codes`` int16 [frames, 10],
``text_ids`` int32 (BOS, payload, EOS) and, with a reference clip, ``speaker_embedding`` float32 [192] and
``ref_codes`` int16 [frames, 10].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from finetuning.sequence import Sequence, build_sequence

SPEAKER_MODES = ("embedding", "reference", "none")


def load_meta(cache: str | Path) -> dict:
    return json.loads((Path(cache) / "meta.json").read_text())


def load_index(cache: str | Path, split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in (Path(cache) / "index.jsonl").read_text().splitlines() if line.strip()]
    return rows if split is None else [row for row in rows if row["split"] == split]


def check_base(meta: dict, config, tokenizer_sha256: str | None) -> None:
    """Refuse a cache made for another codec or tokenizer than the LM's."""
    if meta["codec"]["name"] != config.audio_codec:
        raise ValueError(f"cache codec {meta['codec']['name']!r} is not the LM's {config.audio_codec!r}")
    if meta["tokenizer"]["sha256"] != tokenizer_sha256:
        raise ValueError("the cache was tokenized with another tokenizer than the LM's")


def load_sequence(cache: str | Path, row: dict, config, speaker_mode: str) -> Sequence:
    """The training sequence of one index row. ``speaker_mode``: ``embedding`` (speaker row), ``reference`` (codec
    prefix) or ``none``; a row without a reference clip has no speaker conditioning in any mode."""
    if speaker_mode not in SPEAKER_MODES:
        raise ValueError(f"speaker_mode must be one of {SPEAKER_MODES}")
    with np.load(Path(cache) / f"{row['id']}.npz") as data:
        codes = torch.from_numpy(data["codes"].astype(np.int64))
        text_ids = data["text_ids"].tolist()
        embedding = reference = None
        if speaker_mode == "embedding" and "speaker_embedding" in data:
            embedding = torch.from_numpy(data["speaker_embedding"])
        if speaker_mode == "reference" and "ref_codes" in data:
            reference = torch.from_numpy(data["ref_codes"].astype(np.int64))
    return build_sequence(config, text_ids, codes, speaker_embedding=embedding, reference_codes=reference)
