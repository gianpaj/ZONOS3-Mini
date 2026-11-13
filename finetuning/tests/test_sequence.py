"""Row layout, delay boundaries and loss mask of a training sequence."""

from __future__ import annotations

import torch
from zonos3.tts.delay import shear_up

from finetuning.model import IGNORE
from finetuning.sequence import build_sequence, delayed_audio
from finetuning.tests.conftest import CODEBOOK, N_CB, TEXT_VOCAB

PAD, EOA = CODEBOOK + 1, CODEBOOK
TEXT = [2, 100, 101, 102, 3]


def codes(frames=6, seed=0):
    return torch.randint(0, CODEBOOK, (frames, N_CB), generator=torch.Generator().manual_seed(seed))


def test_delayed_rows_hold_each_frame_then_one_eoa_per_codebook(lm):
    c = codes()
    rows, supervised = delayed_audio(c, lm.config)
    assert rows.shape == (6 + N_CB, N_CB)
    undelayed = shear_up(rows, PAD)
    assert torch.equal(undelayed[:6], c) and (undelayed[6] == EOA).all()
    for j in range(N_CB):
        assert (rows[:j, j] == PAD).all() and not supervised[:j, j].any()  # leading delay triangle
        assert supervised[j : j + 7, j].all() and rows[j + 6, j] == EOA  # frames, then EOA
        assert (rows[j + 7 :, j] == PAD).all() and not supervised[j + 7 :, j].any()  # padding after EOA


def test_targets_are_the_next_row_and_only_audio_is_supervised(lm):
    c = codes()
    seq = build_sequence(lm.config, TEXT, c, speaker_embedding=torch.zeros(8))
    rows, supervised = delayed_audio(c, lm.config)
    n = seq.prompt_rows
    assert n == 1 + 3 + len(TEXT) and len(seq.ids) == n + len(rows) - 1
    # prompt: speaker row, three condition rows, then the text; audio padding everywhere
    assert seq.speaker_rows.keys() == {0} and seq.condition_rows == {1: 4, 2: 4, 3: 4}  # 4 = unknown bucket
    assert seq.ids[4:n, N_CB].tolist() == TEXT and (seq.ids[:4, N_CB] == TEXT_VOCAB).all()
    assert (seq.ids[:n, :N_CB] == PAD).all() and (seq.ids[n:, N_CB] == TEXT_VOCAB).all()
    assert (seq.targets[: n - 1] == IGNORE).all()  # no loss on speaker / condition / text positions
    assert torch.equal(seq.ids[n:, :N_CB], rows[:-1])
    assert torch.equal(seq.targets[n - 1 :], torch.where(supervised, rows, IGNORE))
    # the last text row predicts audio row 0: codebook 0's first code, the other codebooks still in the triangle
    assert seq.targets[n - 1, 0] == c[0, 0] and (seq.targets[n - 1, 1:] == IGNORE).all()
    assert seq.targets[-1, N_CB - 1] == EOA and (seq.targets[-1, : N_CB - 1] == IGNORE).all()
    assert int((seq.targets != IGNORE).sum()) == (6 + 1) * N_CB
    assert not (seq.targets == PAD).any()


def test_reference_prefix_is_unsupervised_context(lm):
    reference = codes(4, seed=3)
    seq = build_sequence(lm.config, TEXT, codes(), reference_codes=reference)
    prefix = 4 + N_CB - 1
    assert seq.speaker_rows is None and seq.condition_rows.keys() == {prefix, prefix + 1, prefix + 2}
    assert torch.equal(shear_up(seq.ids[:prefix, :N_CB], PAD)[:4], reference)
    assert (seq.targets[:prefix] == IGNORE).all() and seq.prompt_rows == prefix + 3 + len(TEXT)
    assert EOA not in seq.ids[:prefix]


def test_no_speaker_conditioning_drops_the_speaker_row(lm):
    seq = build_sequence(lm.config, TEXT, codes())
    assert seq.speaker_rows is None and seq.condition_rows.keys() == {0, 1, 2}
