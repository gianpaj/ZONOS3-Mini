"""python -m finetuning.event_probe clip.wav [more.wav ...]

Scores clips with an AudioSet tagger (MIT/ast-finetuned-audioset-10-10-0.4593, BSD-3-Clause, ~350 MB from the Hub on
first use; CPU). One JSON line per clip: the sigmoid score of the ``speech``, ``laugh``, ``sigh``, ``gasp``, ``cough``
and ``breath`` label groups (the largest score of a group's AudioSet labels).

An automatic screen for "is the event audible": on the codec gate's input / reconstruction pairs and on generated
audio with and without an adapter. The tagger reads the first 10.24 s of a clip, and its scores for short or quiet
events are low even on real recordings, so compare scores between conditions rather than against a threshold.
"""

from __future__ import annotations

import json
import sys

import torch

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
GROUPS = {
    "speech": ("Speech",),
    "laugh": ("Laughter", "Giggle", "Chuckle, chortle", "Belly laugh", "Snicker"),
    "sigh": ("Sigh",),
    "gasp": ("Gasp",),
    "cough": ("Cough", "Throat clearing"),
    "breath": ("Breathing",),
}


class EventProbe:
    def __init__(self):
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.extractor = ASTFeatureExtractor.from_pretrained(MODEL)
        self.model = ASTForAudioClassification.from_pretrained(MODEL).eval()
        ids = {label: index for index, label in self.model.config.id2label.items()}
        self.groups = {name: [ids[label] for label in labels] for name, labels in GROUPS.items()}

    @torch.inference_mode()
    def __call__(self, path: str) -> dict[str, float]:
        import soundfile as sf
        import torchaudio.functional as AF

        wav, rate = sf.read(path, dtype="float32", always_2d=True)
        wave = AF.resample(torch.from_numpy(wav.T.copy()).mean(0), int(rate), 16000)
        inputs = self.extractor(wave.numpy(), sampling_rate=16000, return_tensors="pt")
        scores = self.model(**inputs).logits[0].sigmoid()
        return {name: round(float(scores[index].max()), 4) for name, index in self.groups.items()}


def main(argv=None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print(__doc__, file=sys.stderr)
        return 2
    probe = EventProbe()
    for path in paths:
        print(json.dumps({"clip": path, **probe(path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
