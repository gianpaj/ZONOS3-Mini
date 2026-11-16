"""One training sequence in the LM's prompt layout, with next-row targets.

Rows (``zonos3.tts.plan.prompt_rows``, the layout inference builds)::

    [speaker row | reference prefix] [condition rows] [BOS text EOS] [delayed audio rows]

Audio: frame f of codebook j sits in audio row f + j; after the last frame every codebook carries one EOA (row T + j);
everything else is audio padding. The logits of row r predict the audio columns of row r + 1, so the last text row
predicts audio row 0.

Supervised targets are the codes and the EOA of every codebook. Unsupervised (``IGNORE``): the prompt rows (speaker,
reference prefix, conditions, text), the leading delay triangle (codebook j before its first frame) and the padding
after a codebook's EOA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from zonos3.tts.plan import prompt_rows
from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig, shear, tokens_to_prompt_tokens

from finetuning.model import IGNORE


@dataclass
class Sequence:
    ids: torch.Tensor  # [rows, n_codebooks + 1] int64 inputs
    targets: torch.Tensor  # [rows, n_codebooks] int64, IGNORE where unsupervised
    speaker_rows: dict[int, torch.Tensor] | None
    condition_rows: dict[int, int] | None
    prompt_rows: int  # rows before the first audio row


def delayed_audio(codes: torch.Tensor, config) -> tuple[torch.Tensor, torch.Tensor]:
    """Delayed audio rows [T + n_codebooks, n_codebooks] of ``codes`` [T, n_codebooks] with their EOA, and the
    supervised mask of the same shape."""
    n_cb, frames = config.n_codebooks, len(codes)
    if codes.ndim != 2 or codes.shape[1] != n_cb or frames == 0:
        raise ValueError(f"codes must be [frames > 0, {n_cb}], got {tuple(codes.shape)}")
    if bool(((codes < 0) | (codes >= config.codebook_size)).any()):
        raise ValueError("codes outside the codebook")
    stream = torch.full((frames + n_cb, n_cb), config.audio_pad_id, dtype=torch.long)
    stream[:frames] = codes
    stream[frames] = config.eoa_id
    frame = torch.arange(frames + n_cb)[:, None] - torch.arange(n_cb)[None, :]
    return shear(stream, config.audio_pad_id), (frame >= 0) & (frame <= frames)


def build_sequence(
    config,
    text_ids: list[int],
    codes: torch.Tensor,
    *,
    speaker_embedding: torch.Tensor | None = None,
    reference_codes: torch.Tensor | None = None,
    condition_ids: list[int] | None = None,
) -> Sequence:
    """``config``: the LM's ``Zonos3LMConfig``. ``text_ids``: BOS, payload, EOS. ``condition_ids``: one bucket per
    condition table (default: every table's unknown id, what a request without conditions sends)."""
    if speaker_embedding is not None and reference_codes is not None:
        raise ValueError("a sequence takes a speaker embedding or reference codes, not both")
    n_cb = config.n_codebooks
    builder = TTSPromptBuilder(
        TTSPromptConfig(
            n_codebooks=n_cb,
            codebook_sizes=config.codebook_sizes,
            audio_pad_id=config.audio_pad_id,
            text_vocab=config.text_vocab,
        )
    )
    text = torch.tensor(
        tokens_to_prompt_tokens(
            list(text_ids), n_codebooks=n_cb, audio_pad_id=config.audio_pad_id, text_vocab=config.text_vocab
        ),
        dtype=torch.long,
    )
    if condition_ids is None:
        condition_ids = list(config.condition_bucket_counts)
    if len(condition_ids) != config.condition_count:
        raise ValueError(f"expected {config.condition_count} condition ids, got {len(condition_ids)}")
    prompt = prompt_rows(
        builder,
        text,
        condition_count=config.condition_count,
        reference_codes=None if reference_codes is None else reference_codes.to(torch.long),
        speaker_row=speaker_embedding is not None,
    )
    rows, supervised = delayed_audio(codes.to(torch.long), config)
    n_prompt = len(prompt.input_ids)
    audio = torch.cat((rows[:-1], torch.full((len(rows) - 1, 1), config.text_vocab)), 1)
    ids = torch.cat((prompt.input_ids.to(torch.long), audio))
    targets = torch.full((len(ids), n_cb), IGNORE, dtype=torch.long)
    targets[n_prompt - 1 :] = torch.where(supervised, rows, IGNORE)
    start = prompt.condition_token_start
    return Sequence(
        ids=ids,
        targets=targets,
        speaker_rows=None if speaker_embedding is None else {prompt.speaker_row: speaker_embedding.float()},
        condition_rows={start + i: int(v) for i, v in enumerate(condition_ids)} if start >= 0 else None,
        prompt_rows=n_prompt,
    )
