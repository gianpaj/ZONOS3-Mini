# ZONOS3 Mini LoRA fine-tuning and a nonverbal-vocalization pilot

Decision record for `finetuning/`. Usage is in [finetuning/README.md](../../../../finetuning/README.md).

## Sources

| Source | Revision | Used for |
|---|---|---|
| Zyphra/ZONOS3-Mini (GitHub) | `4df58c814db9163f4bf3f717e9bca49e334a1827` | base of this branch |
| Zyphra/ZONOS3-Mini (Hub) | `727302c1cb61bee229a649d859ebb3e517da1a41` | LM, codec generator, teacher, decoder head, speaker encoder |
| Zyphra/ZONOS3-Mini-hf (Hub) | `e39d30df560125d9c434acd5cb0e83e4c2d64e0d` | read only: `modeling_zonos3.py` |
| Zyphra/ZONOS2_finetuning | `c335b94973ecbffc3d399451d409cad2a2a2e734` | read only: loss, LoRA and config conventions |
| Zyphra/ZAC-1 (Hub) | `0744aff6858a33f83e3946f0d26c050659906b97` | read only: same codec, repackaged |

All three Hub revisions and the trainer commit matched the revisions named in the task on 2026-11-13.

No Mini trainer exists. A search of GitHub and the Hub found launchers, ports and mirrors only; none contains a
teacher-forced forward or a loss for the GDN + MoE backbone. The ZONOS2 trainer targets another architecture (DAC
codes, byte text, no speaker conditioning, `nn.Linear` LoRA on `wq` / `wkv` / `wo` and the head). No code was copied
from it. Two conventions follow it: `up` starts at zero with scale `alpha / rank`, and the loss is one mean
cross-entropy over all codebooks with padding ignored and EOA supervised.

## Architecture-to-gradient map

Release LM: 886,172,520 parameters, bf16 (router balancing biases float32). Trainable: 1,213,568 LoRA parameters at
rank 16 on 44 weights.

| Component | In → out | Dtype | Status | Source |
|---|---|---|---|---|
| Text normalizer, SuperBPE | text → ids (BOS 2, piece + 4, EOS 3) | int | fixed, preprocessing | `zonos3/tokenizer/textnorm.py`, `zonos3/tts/superbpe.py` |
| Dual codec encoder + W2V-BERT teacher (layer 16) | 44.1 kHz mono → codes [T, 10] at 15 Hz, codebook 0 semantic | fp32 weights, bf16 autocast | frozen, preprocessing process only, CUDA only | `zonos3/tokenizer/dual_reference.py`, `zonos3/codec/codec_adapter.py` |
| MECT-B2 speaker encoder | waveform → [192] | fp32 | frozen, preprocessing, CPU | `zonos3/models/mect_speaker.py` |
| Embedders: 10 audio tables [4098, 1024], text [8197, 1024] | ids [R, 11] → summed [R, 1024] | bf16 | frozen | `lm.py` `_MultiEmbedder` |
| `speaker_projection`, 3 `condition_embeddings` | replace the speaker and condition rows | bf16 | frozen | `lm.py` `_replace_rows` |
| Full AttnRes | softmax-weighted mix of the 23 bank sources per sublayer | bf16, scores fp32 | frozen queries and norms | `lm.py` `full_attnres_route` |
| GDN, layers 0-2, 4-6, 8-10 | `in_proj` 1024→2056, causal depthwise conv (kernel 4), delta-rule state [4, 128, 128], `wo` 512→1024 | recurrence fp32 | **LoRA on `in_proj`, `wo`**; conv, `A_log`, `dt_bias`, `out_weight` frozen | `lm.py` `_gdn`; `finetuning/model.py` `_gdn` |
| Global attention, layers 3 and 7 | `wq` 1024→512, packed `wkv` [2, 128, 1024], QK norm, RoPE, head gate, `wo` 512→1024 | bf16 | **LoRA on `wq`, `wkv`, `wo`**; `gater`, `temp` frozen | `lm.py` `_global_attention`; `finetuning/model.py` `_attention` |
| Dense FFN, layer 0 | `w_in` [2, 2048, 1024], `w_out` | bf16 | frozen | `lm.py` `_feed_forward` |
| Latent MoE, layers 1-10 | router 1024→64, top-2; `latent_down` 1024→256; experts `gate_up_proj` [64, 3072, 256], `down_proj` [64, 256, 1536]; `latent_up` 256→1024 | bf16, routing weights fp32 | **LoRA on `latent_down`, `latent_up`**; router and experts (755 M) frozen | `lm.py` `_feed_forward`; `finetuning/model.py` `_feed_forward` |
| Head | `out_norm`, `multi_output` 1024→40980, softcap 15·tanh(x/15) → logits [R, 10, 4098] | bf16 | frozen | `lm.py` `forward` |
| iSTFT decoder `keep2_plain` | codes → 44.1 kHz PCM | fp32 | frozen, codec gate and inference only | `zonos3/tokenizer/matched_codec.py` |

The top-2 expert choice is a hard selection. Gradients pass through the softmax over the two chosen router logits and
through the chosen experts, not through the choice.

## One example, end to end

`libritts_r_1284_1180_000004_000004`: 7.12 s, 24 kHz mono, "Instead of shoes, the old man wore boots with turnover
tops and his blue coat had wide cuffs of gold braid."

1. Text: normalizer (unchanged here), SuperBPE → 36 ids including BOS and EOS.
2. Audio: resampled to 44.1 kHz, no peak scaling → 107 frames × 10 codes, each in [0, 4096).
3. Speaker: a different utterance of speaker 1284 → a 192-d embedding (and, for reference mode, its codes).
4. Rows, embedding mode: row 0 speaker, rows 1-3 conditions with the unknown buckets (10, 10, 30), rows 4-39 text,
   then 116 delayed audio rows. 156 input rows, 40 of them prompt.
5. Targets: row 39 (the EOS text row) predicts audio row 0. Each codebook contributes 107 codes and one EOA:
   1,080 supervised positions. The 45 positions of the leading delay triangle and the padding after each EOA are
   ignored.
6. Reference mode: the reference clip's 123 frames become a 132-row delayed prefix in front of the condition rows,
   unsupervised. 287 rows, the same 1,080 targets.
7. Loss: mean cross-entropy over the 1,080 positions. The base model scores 6.69 nats per code here (codebook 0:
   4.23; residual codebooks 5.9-7.7; uniform is 8.32).

## Decisions

**A separate training forward, checked against the oracle.** `finetuning/model.py` reuses the `Zonos3LM` module, its
strict loader and its pure helpers, and reimplements the GDN, attention and MoE sublayers without a cache.
`zonos3.reference` is untouched, so the oracle and its tests stay as shipped.
Rejected: removing `@torch.no_grad()` from the oracle (its per-(token, expert) dispatch syncs the GPU twice per token
per layer, and a shared refactor would put the oracle at risk); the Transformers export (also frozen and `no_grad`, and
its chunked GDN writes slices in place).

**GDN recurrence token by token in float32**, as the oracle runs it. Rejected for now: the chunked rule. It is the
route to long clips, and it needs its own parity gate.

**Experts dispatched per routed expert.** Tokens that chose expert `e` go through it in one call and are summed back
with `index_add`. Each term is cast to the activation dtype before the sum, as in the oracle.

**The adapter is a native `zonos-accent-lora/1` file.** The trainable module answers the hook `AccentState` serves
(`linear(module, x, y)`) under the checkpoint's module names, and saves `down` [rank, in], `up` [out, rank] and the
weight shape with `alpha / rank` folded into `up`. Packed weights (`wkv`) count as their flat product, which is how
the oracle applies them. Rejected: PEFT (it wraps `nn.Linear` modules; the packed weights are not, and the result
would need an exporter).

**Targets: mixer and latent projections.** The router stays frozen, so the balancing biases keep their meaning. Expert
factors (`/2`) and new vocabulary rows (`/3`) were not needed.

**Loss.** Mean cross-entropy over all codes and EOA positions, every codebook weighted alike, on softcapped logits.
Upstream's weighting is not public; this is the ZONOS2 trainer's definition.

**EOA and delay boundaries are recovered, not guessed.** Inference stops on a stop token and aligns it by codebook
index (`synth.py` `generate_rows`), and `delay.py` says the leading triangle is unsupervised. The base model confirms
the layout: on three clips it puts 0.81, 0.98 and 1.00 on codebook 0's EOA at the target row and at most 0.17 on any
other row. Shifting the targets by one row raises the loss from 6.7 to 11.6-11.8; a wrong transcript raises codebook
0's loss from 4.23 to 6.87.

**Condition rows carry the unknown bucket.** Loudness, fmax and speaking-rate edges are in `params.json`, the
measurement code is not. A request without conditions sends the same rows.

**Speaker conditioning from a separate clip.** The embedding row is the default; the reference prefix is available
(`--speaker-mode reference`). Upstream trained the prefix with a separate clip and added the embedding in a short
later stage (Zyphra blog).

**Event tags are plain text.** `[laugh]` tokenizes to `[`, `la`, `ug`, `h`, `]`. The request normalizer keeps the
tags; preprocessing fails if it ever does not.

**Preprocessing is its own process.** The codec encoder is CUDA only and peaks at 5.64 GiB; the trainer never loads
it.

**Left out.** Padded batches, resume, LoRA dropout, LoRA+ and Muon, computed condition buckets.

## Verification

RTX 4060 Ti 16 GB, torch 2.9.1+cu128, 2026-11-13. The Studio's `zonos3-server` held 4.56 GiB of the GPU throughout
and was not touched.

| Gate | Result |
|---|---|
| Strict load | `Zonos3LM.load` (`load_state_dict(strict=True)`), no key or shape mapping needed |
| Base logits, training forward vs oracle, float32, real model, 2 clips (180 and 185 rows) | max abs 4.0e-5 and 5.0e-5, mean KL 2e-9 and 3e-9 nats, top-1 agreement 100 % |
| Same in bf16, both against the float32 oracle | training forward: mean KL 0.0025 and 0.0060; the bf16 oracle itself: 0.0044 and 0.0046 |
| Finite loss on real encoded clips | yes (6.02 on the first step of the smoke run) |
| First-step gradients | all 44 `up` factors nonzero and finite; all 44 `down` gradients exactly zero, as `up` starts at zero |
| Base parameters after training | none has `requires_grad` or a gradient (asserted at the end of every run) |
| Backward seam, tiny model (`tests/test_model.py`) | directional derivatives match central differences within 2 %; layer checkpointing gives the same loss and gradients; an optimizer step lowers the loss and leaves the base unchanged |
| Fresh process, adapter reloaded for training vs logits stored before saving | bf16: identical (max abs 0.0) |
| Fresh process, saved file served by `load_adapter` + `AccentState` through the oracle vs the reloaded adapter, float32 | max abs 2.8e-5, top-1 agreement 100 % |
| Adapter vs base on the same sequence (smoke adapter, 3 steps) | mean KL 0.245 nats: the adapter is applied |
| Synthesis with the saved adapter | `python -m zonos3.reference --accent-adapters` and `finetuning.generate` produce audio that differs from the base at the same seed |
| Encoder determinism on this machine | 72 of 72 clips re-encoded to identical codes, embeddings and text ids |
| Encoder vs the shipped voice codes (2 clips) | codebook 0: 96-97 % equal; residual codebooks 65-90 %. Upstream documents 97-98 % and 67-96 % across runtimes (`servers/mlx/README.md`) |
| Tests | `pytest finetuning/tests`: 18 passed (CPU); `model/tests/test_reference_lm.py` still passes, `model/` is unchanged |

## Memory

Measured peaks of the training process (`torch.cuda.max_memory_allocated` / `max_memory_reserved`, and the process
row of `nvidia-smi`), bf16 base, rank 16, default targets, one sequence per forward. Concurrent GPU process: the
Studio's `zonos3-server`, 4,560 MiB.

| Sequence | Rows | Options | Allocated | Reserved | `nvidia-smi` | s/step |
|---|---|---|---|---|---|---|
| pilot clips (up to 8 s), embedding | 169 | | 2,785 MiB | 4,432 MiB | 4,596 MiB | 0.33 |
| 24 s clip, embedding | 398 | | 4,235 | 5,560 | 5,724 | 1.24 |
| 24 s clip, embedding | 398 | `--checkpoint-layers` | 2,056 | 3,804 | 3,968 | 1.92 |
| 24 s clip + 10 s reference prefix | 556 | | 5,241 | 6,218 | 6,382 | 1.68 |
| 24 s clip + 10 s reference prefix | 556 | `--checkpoint-layers` | 2,190 | 3,850 | 4,014 | 2.61 |
| 24 s clip, embedding, float32 base | 398 | `--dtype f32` | 6,314 | 6,950 | 7,108 | 1.18 |
| 24 s clip, rank 64, all 10 linear target kinds (8,199,936 parameters) | 398 | | 4,330 | 5,614 | 5,778 | 1.23 |

Itemized, 398 rows, bf16, rank 16:

| Item | MiB | Kind |
|---|---|---|
| Frozen weights (886 M, bf16) plus adapter | 1,695 | measured after load |
| Adapter weights / gradients / AdamW moments | 4.6 / 4.6 / 9.3 | computed from 1,213,568 float32 parameters |
| Logits, 398 × 10 × 4,098, bf16 and the float32 copy for the loss | 93 | computed |
| Activations and workspaces | 2,433 (6.1 per row) | peak minus the items above |
| Allocator cache above the peak | 1,325 | reserved minus allocated |
| CUDA context and libraries | 164 | `nvidia-smi` minus reserved |

Projection, not measured: a 30 s clip with an embedding is about 500 rows, so about 4.8 GiB allocated without
checkpointing. Preprocessing peaks higher than any training run here: 5.64 GiB for the codec encoder.

The Studio has been sampled at 10.9 GiB after a speech edit, when its own reference encoder is loaded. Next to that,
5.4 GiB of the card is left: the pilot-length run and the checkpointed runs fit, the 24 s runs without checkpointing
and preprocessing do not. Every run here happened while the Studio sat at 4.56 GiB.

## Pilot: laugh, sigh, gasp

The first domain is human nonverbal vocalization, three labels, with plain speech mixed in.

**Data.** 72 clips, 297 s, speaker-grouped splits (every fourth speaker of each source held out; no speaker or source
recording on both sides). Sources, licences and per-clip credits: [finetuning/pilot/PROVENANCE.md](../../../../finetuning/pilot/PROVENANCE.md);
rows: [finetuning/pilot/manifest.jsonl](../../../../finetuning/pilot/manifest.jsonl). The audio is not in the
repository.

| Event | Train | Held-out | Source |
|---|---|---|---|
| laugh | 14 | 6 | VocalSound (CC BY-SA 4.0), FSD50K (CC0 / CC BY 3.0 clips) |
| sigh | 16 | 5 | VocalSound, FSD50K |
| gasp | 8 | 3 | FSD50K |
| speech | 16 | 4 | LibriTTS-R test.clean (CC BY 4.0) |

26 training speakers and 8 held-out. Every nonverbal clip is an isolated event whose text is the bare tag (`[laugh]`,
`[sigh]`, `[gasp]`); no clip mixes a tag with words. The VocalSound and FSD50K speakers have no speech, so their
reference clip is another nonverbal clip of the same speaker; LibriTTS-R references are speech. FSD50K "speakers" are
Freesound uploaders, a proxy.

Dropped before training: 14 AMI meeting clips with a transcribed laugh, the only source with a tag inside speech. They
are quiet (RMS 0.003-0.05) and the tagger hears breath in the originals, not laughter (laugh score 0.00).

**Codec gate.** Scores are the tagger's for the clip's own label, median, original → reconstruction.

| Source, event | Clips | Log-mel L1 | Tagger score |
|---|---|---|---|
| LibriTTS-R speech | 20 | 0.87 | 0.83 → 0.85 |
| VocalSound sigh | 16 | 0.72 | 0.71 → 0.52 |
| FSD50K sigh | 5 | 0.70 | 0.61 → 0.49 |
| FSD50K gasp | 11 | 0.79 | 0.60 → 0.36 |
| VocalSound laugh | 16 | 0.79 | 0.06 → 0.10 |
| FSD50K laugh | 4 | 1.05 | 0.12 → 0.05 |

The codec reconstructs these sounds with a spectral error in the range of speech. The tagger still recognizes sighs
and gasps after reconstruction, with lower scores. It scores laughter low on the original recordings too, so it cannot
confirm or rule out laughter anywhere in this pilot. No class was excluded on this evidence.

**Overfit diagnostic** (plumbing fixture, 10 training clips relabelled `overfit`, 50 steps): mean loss 5.74 → 4.16,
all 10 clips improved.

**Pilot run.** 54 training clips, rank 16, `--lr 5e-4`, bf16, embedding mode, one sequence per step, 0.33 s per step.

| Held-out loss (nats per code) | Clips | Base | 100 steps | 200 steps |
|---|---|---|---|---|
| gasp | 3 | 6.71 | 6.36 | 6.59 |
| laugh | 6 | 6.95 | 6.58 | 6.77 |
| sigh | 5 | 6.28 | 5.96 | 6.09 |
| speech | 4 | 6.50 | 6.24 | 6.45 |
| all | 18 | 6.62 | 6.30 (18 of 18 clips lower) | 6.48 (17 of 18) |

The training loss at 200 steps is 5.57 → 4.44, so the adapter is memorizing by then; the 100-step adapter is the pilot
result. The step count was chosen on the held-out clips, which makes the 100-step column slightly optimistic. The
speech rows improve as well, which points to adaptation to the recordings as much as to the tags.

**Samples.** 96 clips: 4 held-out speaker embeddings (2 VocalSound, from a nonverbal clip; 2 LibriTTS-R, from speech)
× 6 texts × seeds 0 and 1 × base and adapter. Mean tagger score of the expected label, base → adapter, and the
number of same-seed pairs where the adapter scores higher.

| Text | Embedding from | Expected label | Speech score |
|---|---|---|---|
| `[sigh]` | speech | sigh 0.01 → 0.48 (4/4) | 0.80 → 0.35 |
| `[sigh]` | nonverbal clip | sigh 0.36 → 0.16 (2/4) | 0.02 → 0.03 |
| `[sigh] I suppose we have to start over.` | speech | sigh 0.04 → 0.32 (3/4) | 0.75 → 0.78 |
| `[sigh] I suppose we have to start over.` | nonverbal clip | sigh 0.01 → 0.30 (3/4) | 0.59 → 0.55 |
| `[laugh]` | speech | laugh 0.000 → 0.002 (4/4) | 0.71 → 0.54 |
| `[laugh]` | nonverbal clip | laugh 0.000 → 0.072 (3/4) | 0.01 → 0.01 |
| `I can't believe you did that. [laugh]` | speech | laugh 0.001 → 0.002 (4/4) | 0.66 → 0.57 |
| `I can't believe you did that. [laugh]` | nonverbal clip | laugh 0.001 → 0.000 (1/4) | 0.62 → 0.21 |
| `[gasp]` | speech | gasp 0.01 → 0.03 (3/4) | 0.74 → 0.72 |
| `[gasp]` | nonverbal clip | gasp 0.11 → 0.12 (2/4) | 0.05 → 0.04 |
| `The quick brown fox jumps over the lazy dog.` | speech | speech 0.72 → 0.76 (3/4) | |
| `The quick brown fox jumps over the lazy dog.` | nonverbal clip | speech 0.25 → 0.75 (3/4) | |

What this supports, with four pairs per row and an automatic tagger:

- `[sigh]` moves toward a sigh with a clean-speech speaker embedding, alone and at the start of a sentence, and the
  sentence is still scored as speech.
- `[gasp]` shows no effect. It had 8 training clips.
- `[laugh]` is unresolved: the tagger scores real laughter at 0.06-0.12.
- Plain speech is still scored as speech with the adapter.
- A speaker embedding taken from a nonverbal clip is a confound: with it the base model already produces non-speech
  audio for a bare tag (speech score 0.01-0.05), and less speech-like audio for a plain sentence (0.25). A clean-speech
  embedding is the better condition for telling a learned tag from a leaking reference.
- Two adapter samples and one base sample, all with a nonverbal-clip embedding, ran to the 150-frame budget without an
  EOA. With that embedding the adapter also lowers the speech score of the laugh sentence (0.62 → 0.21).

What it does not show: that a listener hears the events, that the tags work inside a sentence in general (no training
clip has one), or anything about other speakers, languages or sound classes.

## Open

- **Listening.** Nobody has listened to the gate pairs or the samples. The tagger scores are a screen.
- **Native server route.** `tools/convert_gguf.py accents` packs the pilot adapter into an `accents.gguf` (1 adapter,
  92 tensors, format `zonos-accent-lora/1`, rank 16). No native server has loaded it: the only built binary is the
  live Studio's. The released GGUF backbones are quantized, a separate variable. Next step for review: build
  `zonos3-cli` in this worktree and run it on the CPU with `--accents` and `--accent nv`.
- **Long clips.** The chunked GDN rule, then padded batches.
- **Speaker sidecars.** The PyTorch MECT encoder's embeddings of the shipped voice clips differ from the shipped
  `.embedding.npy` files (cosine 0.49-0.90 on four voices). The trainer uses the PyTorch encoder, as the Python servers
  do. Unexplained; the native `spk-encoder-cli` was not built to compare.
