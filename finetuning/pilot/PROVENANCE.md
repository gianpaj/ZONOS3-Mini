# Pilot data: sources, licences and selection

[manifest.jsonl](manifest.jsonl) lists the 72 clips of the nonverbal pilot. `audio` and `ref_audio` are relative to a
folder that holds the downloaded clips; the audio is not in this repository. Each row's `provenance` names the dataset
and the original file.

Clips are byte-exact copies of the source files: nothing was resampled, normalized or re-encoded before
`finetuning.preprocess`. They were chosen by rules over metadata. Nobody listened to them.

Splits: within each source, the sorted speaker ids' every fourth entry (index 3, 7, 11, ...) is `heldout`. The
LibriTTS-R reference clips appear only as `ref_audio`.

## VocalSound: laugh, sigh (32 clips, 16 subjects)

- https://github.com/YuanGongND/vocalsound, archive `vs_release_44k.zip`.
- Licence: CC BY-SA 4.0. The repository README states: "The VocalSound dataset is licensed under a Creative Commons
  Attribution-ShareAlike 4.0 International License."
- Attribution: Yuan Gong, Jin Yu, James Glass, "VocalSound: A Dataset for Improving Human Vocal Sounds Recognition",
  ICASSP 2022.
- Selection: subjects with native language `EN`, health condition `no`, gender male or female, and laughter, sigh and
  cough takes of 1-8 s; 8 evenly spaced subjects per gender in id order. A clip's `ref_audio` is the same subject's
  other class.
- The files keep their crowdsourced sample rates: 27 at 48 kHz, 4 at 44.1 kHz, 1 at 16 kHz.
- ShareAlike: what CC BY-SA requires of a model trained on these clips was not assessed.

## FSD50K: gasp, and laugh and sigh from the same uploaders (20 clips, 9 uploaders)

- https://zenodo.org/records/4060432; clips fetched from the mirror https://huggingface.co/datasets/Fhrozen/FSD50k.
- Licence: per clip, set by the Freesound uploader. Only CC0 1.0 and CC BY 3.0 clips are used; CC BY-NC and Sampling+
  clips are excluded. The collection is CC BY 4.0 (Fonseca et al., "FSD50K: An Open Dataset of Human-Labeled Sound
  Events", IEEE/ACM TASLP 2022).
- Selection: clips whose labels are within {Gasp, Sigh, Laughter, Giggle, Chuckle_and_chortle, Human_voice, Breathing,
  Respiratory_sounds} with exactly one target class, no group words in the title, 1-8 s; per uploader up to 2 gasps,
  1 laugh and 1 sigh; uploaders kept only with a gasp and at least one other class.
- `speaker` is the Freesound uploader, a proxy: nothing guarantees one voice per uploader. Labels are clip-level.
- The per-clip licences come from the FSD50K metadata and were not re-checked on freesound.org.

Credits (Freesound id, title, uploader, licence):

| Id | Title | Uploader | Licence | Event |
|---|---|---|---|---|
| [401333](https://freesound.org/s/401333/) | Man gasp of breath | ckvoiceover | CC0 1.0 | gasp |
| [401332](https://freesound.org/s/401332/) | Evil Laugh | ckvoiceover | CC0 1.0 | laugh |
| [19742](https://freesound.org/s/19742/) | POW07-E01-Voice.mp3 | Freed | CC BY 3.0 | gasp |
| [19695](https://freesound.org/s/19695/) | POW01-E01-Voice.mp3 | Freed | CC BY 3.0 | sigh |
| [31356](https://freesound.org/s/31356/) | gasp01.wav | FreqMan | CC BY 3.0 | gasp |
| [31357](https://freesound.org/s/31357/) | gasp02.wav | FreqMan | CC BY 3.0 | gasp |
| [42841](https://freesound.org/s/42841/) | laugh 1.wav | FreqMan | CC BY 3.0 | laugh |
| [333396](https://freesound.org/s/333396/) | Gasp 4.wav | jayfrosting | CC0 1.0 | gasp |
| [333398](https://freesound.org/s/333398/) | Gasp 2 with wonder.wav | jayfrosting | CC0 1.0 | gasp |
| [333400](https://freesound.org/s/333400/) | Laugh 2 good natured.wav | jayfrosting | CC0 1.0 | laugh |
| [363631](https://freesound.org/s/363631/) | sonidos para el final_stems_Inhalar.wav | nitsuga94 | CC0 1.0 | gasp |
| [363646](https://freesound.org/s/363646/) | sonidos para el final_stems_Suspiro de terror.wav | nitsuga94 | CC0 1.0 | sigh |
| [65620](https://freesound.org/s/65620/) | Inhale_10115a.wav | otherthings | CC0 1.0 | gasp |
| [65621](https://freesound.org/s/65621/) | Inhale_10115b.wav | otherthings | CC0 1.0 | gasp |
| [65619](https://freesound.org/s/65619/) | Exhale_10115a.wav | otherthings | CC0 1.0 | sigh |
| [416510](https://freesound.org/s/416510/) | Male gasp | pfranzen | CC BY 3.0 | gasp |
| [417750](https://freesound.org/s/417750/) | Sad male sigh :( | pfranzen | CC BY 3.0 | sigh |
| [343909](https://freesound.org/s/343909/) | breath HUUUAH.wav | Reitanna | CC0 1.0 | gasp |
| [215339](https://freesound.org/s/215339/) | evil laugh.wav | Reitanna | CC0 1.0 | laugh |
| [215380](https://freesound.org/s/215380/) | male sigh.wav | Reitanna | CC0 1.0 | sigh |

## LibriTTS-R: read speech (20 clips and 10 reference clips, 10 speakers)

- https://www.openslr.org/141/, read from https://huggingface.co/datasets/mythicinfinity/libritts_r (`test.clean`).
- Licence: CC BY 4.0, as stated on the OpenSLR page.
- Attribution: Koizumi et al., "LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus", Interspeech 2023; derived
  from LibriTTS, LibriSpeech and LibriVox recordings.
- Selection: per speaker in numeric order, the first 2 utterances of 3-8 s as targets and the first other utterance of
  4-10 s as the reference, preferring a different chapter. 24 kHz mono.
- `text` is `text_normalized`. `source` is `speaker_chapter`.

## Checked and not used

- AMI Meeting Corpus (CC BY 4.0): 14 clips of meeting speech ending in a transcribed laugh were collected and dropped.
  They are quiet and an AudioSet tagger hears breath in them, not laughter.
- NonverbalTTS (annotations CC BY-NC-SA), Expresso (CC BY-NC 4.0), ESC-50 (CC BY-NC): non-commercial terms.
- Nonspeech7k (CC BY 4.0): no subject ids and no gasp class.
- JNV corpus: licence terms could not be read from its page.
