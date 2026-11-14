# LoRA fine-tuning for ZONOS3 Mini

A single-GPU LoRA trainer for the ZONOS3 Mini LM. It reads waveform/text pairs, tokenizes them with the model's frozen
front ends, trains low-rank factors with a teacher-forced forward and saves them as a `zonos-accent-lora/1` adapter,
the format the reference synthesizer and the servers load as an accent.

Scope: a working training path and a small nonverbal-vocalization pilot. It is not a recipe for production quality,
and it does not reproduce Zyphra's pretraining, DPO or GRPO. Status, a data-flow diagram, the file map and next steps
are in [OVERVIEW.md](OVERVIEW.md). Design rationale, measurements and pilot results are in
[the Agent Note](../.agents/notes/implemented/finetuning/2026-11-13-zonos3-mini-lora-finetuning.md).

## Setup

```bash
git submodule update --init third_party/fusealign-multilingual   # a path dependency of the zonos3 package
uv sync --locked --package zonos3
```

Commands below run from the repository root with `uv run --package zonos3 python -m finetuning.<tool>` (or
`.venv/bin/python -m finetuning.<tool>`).

First use downloads from `Zyphra/ZONOS3-Mini` into the Hugging Face cache, about 11 GB: the LM (1.8 GB), the codec
generator (8.0 GB), its semantic teacher (0.8 GB), the `keep2_plain` decoder head (0.5 GB) and the speaker encoder
(50 MB). `finetuning.event_probe` adds an AudioSet tagger (0.35 GB).

## Pipeline

| Step | Tool | Needs |
|---|---|---|
| Describe the data | a JSONL manifest ([manifest.py](manifest.py)) | |
| Tokenize | `finetuning.preprocess` | CUDA, 5.7 GiB peak for the codec encoder |
| Listen to what the codec keeps | `finetuning.codec_gate`, `finetuning.event_probe` | CUDA or CPU |
| Train | `finetuning.train` | CUDA; see [Memory](#memory) |
| Verify the saved adapter | `finetuning.check` | |
| Held-out loss | `finetuning.evaluate` | |
| Generate | `finetuning.request`, then `finetuning.generate` or `python -m zonos3.reference` | |

```bash
P="uv run --package zonos3 python -m"

# 1. Tokenize: text normalizer + SuperBPE, matched dual codec + semantic teacher, MECT speaker embedding.
$P finetuning.preprocess --manifest data/pilot.jsonl --cache work/cache

# 2. Codec gate: input / reconstruction pairs, report.json and a listening checklist.
$P finetuning.codec_gate --cache work/cache --out work/gate
$P finetuning.event_probe work/gate/*.wav > work/gate/probe.jsonl

# 3. Train. The default is one step; up to 50 steps is the capped diagnostic; more needs --long.
$P finetuning.train --cache work/cache --out work/adapters --name nv                    # smoke
$P finetuning.train --cache work/cache --out work/adapters --name nv --steps 100 --long

# 4. Fresh-process checks: training forward against the inference oracle, and the saved adapter.
$P finetuning.check --cache work/cache --dtype f32
$P finetuning.check --cache work/cache --dtype f32 --adapter work/adapters/nv.pt --examples 0

# 5. Teacher-forced loss per event label, base against adapter.
$P finetuning.evaluate --cache work/cache --split heldout --adapter work/adapters/nv.pt

# 6. Generate with the adapter served as accent "nv" (drop --accent for the base model).
$P finetuning.request --text "[sigh]" --accent nv --cache work/cache --example SOME_ID -o work/req/sigh.json
$P finetuning.generate --accent-adapters work/adapters --out-dir work/wav work/req/*.json
# the upstream CLI takes the same request and adapter directory, one request per process:
$P zonos3.reference --model hf://Zyphra/ZONOS3-Mini/lm/zonos3_mini --phoneme-rows '' --dtype bf16 \
    --accent-adapters work/adapters --request work/req/sigh.json -o work/out.wav
```

Tests: `uv run --package zonos3 pytest finetuning/tests` (CPU, a tiny random model).

## Manifest

One JSON object per line; [manifest.py](manifest.py) documents every field and validates them.

```json
{"id": "spk1_laugh_01", "audio": "clips/spk1_laugh_01.wav", "text": "[laugh]", "language": "en_us",
 "event": "laugh", "speaker": "spk1", "source": "session_07", "split": "train",
 "license": "CC BY 4.0", "provenance": "https://… original file name", "ref_audio": "clips/spk1_read_03.wav"}
```

- `text` is what the model reads. Event tags such as `[laugh]` are a convention the adapter learns: the base model has
  no such control and reads the brackets as text. The tags tokenize to existing SuperBPE pieces (`[`, `la`, `ug`, `h`,
  `]`), so no vocabulary row is added. Preprocessing fails if the text normalizer alters a tag.
- `event` labels a row for evaluation. It never reaches the model.
- `split`: `train` and `heldout` may not share a speaker or a source recording. `overfit` marks a plumbing fixture that
  is trained and evaluated on the same rows. `manifest.assign_splits` assigns grouped splits.
- `ref_audio` is a different clip of the same speaker. Its MECT embedding (or its codec codes) conditions the
  sequence; the target clip is never its own reference unless the row sets `self_reference` for a diagnostic.
  A row without `ref_audio` trains without speaker conditioning.

## What a training sequence is

```
[speaker row | reference prefix] [3 condition rows] [BOS text EOS] [delayed audio rows]
```

The rows come from `zonos3.tts.plan.prompt_rows`, the function inference uses. Frame `f` of codebook `j` sits in audio
row `f + j`; one EOA follows each codebook's last frame; the logits of a row predict the audio columns of the next row.
The condition rows carry the "unknown" bucket of each table, which is what a request without conditions sends.

Loss: mean cross-entropy over every codec code and every EOA, all ten codebooks weighted alike, on the softcapped
logits. Not supervised: the speaker row, reference prefix, condition and text rows, the leading delay triangle and the
padding after a codebook's EOA. [sequence.py](sequence.py) builds this and `tests/test_sequence.py` pins it.

Each forward is one sequence from row 0 with fresh GDN state and no attention cache, so nothing crosses examples.
`--grad-accum N` averages N sequences per optimizer step.

## What is trained

LoRA factors on `attention.in_proj`, `attention.wq`, `attention.wkv`, `attention.wo`, `feed_forward.latent_down` and
`feed_forward.latent_up` of every layer that has them (`--targets` picks any entry of
`zonos3.models.accent_format.TARGETS`). Rank 16 gives 1,213,568 trainable parameters on 44 weights. The base LM
(886 M parameters, bf16), the router, the experts, the embeddings and the output head stay frozen, as do the codec, the
teacher, the decoder and the speaker encoder, which the training process never loads.

`--rank 16 --alpha 16 --lr 5e-4`, AdamW and gradient clipping at 1.0 are starting values that worked in the pilot, not
tuned settings.

## Adapter files

`OUT/<name>.pt` is a `zonos-accent-lora/1` file with `alpha / rank` folded into the `up` factors, so it is served at
accent strength 1. `OUT/<name>.json` records the arguments, base and cache identity, trainable tensors, losses and
memory. The reference synthesizer loads the directory as is (`python -m zonos3.reference --accent-adapters OUT`, or
`finetuning.generate` for several requests in one process).

For the ggml server, `uv run --package zonos3-tools python tools/convert_gguf.py accents OUT -o accents.gguf` packs the
directory; the converter accepts the default targets. Serving that file is not validated here, and the released GGUF
backbones are quantized, which is a separate quality variable.

Resuming a run is not supported: the optimizer state is not saved.

## Memory

Measured on an RTX 4060 Ti 16 GB next to a running Studio, base weights in bf16, rank 16, default targets, batch of
one sequence. "Rows" is the whole sequence: prompt plus audio frames plus 9.

| Sequence | Rows | Options | Peak allocated | Peak reserved | Process in `nvidia-smi` | s/step |
|---|---|---|---|---|---|---|
| clips up to 8 s, speaker embedding | 169 | | 2.72 GiB | 4.33 GiB | 4.49 GiB | 0.33 |
| 24 s clip, speaker embedding | 398 | | 4.14 GiB | 5.43 GiB | 5.59 GiB | 1.24 |
| 24 s clip, speaker embedding | 398 | `--checkpoint-layers` | 2.01 GiB | 3.71 GiB | 3.88 GiB | 1.92 |
| 24 s clip + 10 s reference prefix | 556 | `--speaker-mode reference` | 5.12 GiB | 6.07 GiB | 6.23 GiB | 1.68 |
| 24 s clip + 10 s reference prefix | 556 | `--speaker-mode reference --checkpoint-layers` | 2.14 GiB | 3.76 GiB | 3.92 GiB | 2.61 |
| 24 s clip, speaker embedding | 398 | `--dtype f32` | 6.17 GiB | 6.79 GiB | 6.94 GiB | 1.18 |
| 24 s clip, rank 64 on every linear target (8.2 M) | 398 | | 4.23 GiB | 5.48 GiB | 5.64 GiB | 1.23 |

The frozen bf16 weights take 1.66 GiB. Adapter weights, gradients and AdamW state together are under 20 MiB at rank
16. The logits of 398 rows take 93 MiB. The rest is activations, about 6.1 MiB per row without checkpointing.

Activations grow linearly with rows: every GDN layer keeps its per-token recurrent state for the backward pass.
`--checkpoint-layers` recomputes one layer at a time instead. Preprocessing is the larger consumer: the codec encoder
peaks at 5.64 GiB.

## Limits

- The GDN recurrence runs token by token in Python. It matches the oracle and is fast enough for short clips
  (0.33 s per step up to 169 rows, 1.24 s at 398); large datasets need the chunked rule.
- One sequence per forward; no padded batches.
- bf16 training logits differ from float32 by about as much as the bf16 oracle does. `finetuning.check --dtype bf16
  --oracle-dtype f32` prints both gaps.
- The loss weighting, optimizer settings and the condition-row choice are this trainer's. Upstream's training
  configuration is not public.
