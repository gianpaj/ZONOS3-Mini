"""Manifest validation, grouped splits and event-tag survival through text normalization."""

from __future__ import annotations

import json

import pytest

from finetuning.manifest import Row, assign_splits, load_manifest, validate
from finetuning.preprocess import normalize_text


def row(id, split="train", speaker=None, source=None, audio=None, **extra) -> Row:
    return Row(
        id=id,
        audio=audio or f"/data/{id}.wav",
        text="Hello [laugh] there.",
        language="en_us",
        event="laugh",
        split=split,
        license="CC BY 4.0",
        provenance="unit test",
        speaker=speaker,
        source=source,
        **extra,
    )


def test_a_speaker_or_source_on_both_sides_is_refused():
    validate([row("a", "train", speaker="s1"), row("b", "heldout", speaker="s2")])
    with pytest.raises(ValueError, match="speaker:s1"):
        validate([row("a", "train", speaker="s1"), row("b", "heldout", speaker="s1")])
    with pytest.raises(ValueError, match="source:rec9"):
        validate([row("a", "train", speaker="s1", source="rec9"), row("b", "heldout", speaker="s2", source="rec9")])
    # an overfit fixture reuses its rows on purpose and is exempt
    validate([row("a", "overfit", speaker="s1"), row("b", "heldout", speaker="s1")])


def test_reference_rules():
    with pytest.raises(ValueError, match="self_reference"):
        validate([row("a", ref_audio="/data/a.wav")])
    validate([row("a", ref_audio="/data/a.wav", self_reference=True)])
    with pytest.raises(ValueError, match="heldout target"):
        validate([row("a", "train", speaker="s1", ref_audio="/data/b.wav"), row("b", "heldout", speaker="s2")])


def test_ids_languages_and_splits_are_checked():
    for bad in (row("a b"), row("a", split="dev"), Row(**{**row("a").to_json(), "language": "en"})):
        with pytest.raises(ValueError):
            validate([bad])
    with pytest.raises(ValueError):
        validate([row("a"), row("a")])


def test_assign_splits_keeps_linked_rows_together():
    rows = [row(f"r{i}", speaker=f"s{i // 2}", source=f"rec{i // 4}") for i in range(16)]
    rows.append(row("fixture", "overfit", speaker="s0"))
    out = assign_splits(rows, 0.25, seed=3)
    assert out == assign_splits(rows, 0.25, seed=3)
    side = {}
    for r in out[:-1]:
        assert side.setdefault(r.source, r.split) == r.split  # 4 rows per source, two speakers each
    assert sorted(side.values()).count("heldout") == 1 and out[-1].split == "overfit"


def test_load_manifest_resolves_paths_and_requires_fields(tmp_path):
    doc = {**row("a", audio="clips/a.wav").to_json(), "ref_audio": "clips/ref.wav"}
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps(doc) + "\n")
    (loaded,) = load_manifest(path)
    assert loaded.audio == str(tmp_path / "clips/a.wav") and loaded.ref_audio == str(tmp_path / "clips/ref.wav")
    del doc["license"]
    path.write_text(json.dumps(doc) + "\n")
    with pytest.raises(ValueError, match="license"):
        load_manifest(path)


def test_event_tags_survive_the_request_normalizer():
    from zonos3.tokenizer.textnorm import TTSTextNormalizer

    normalizer = TTSTextNormalizer()
    for text in ("[laugh]", "Well, [sigh] I paid $5 for it.", "[gasp] You scared me!"):
        r = Row(**{**row("a").to_json(), "text": text})
        normalized = normalize_text(r, normalizer)
        assert all(tag in normalized for tag in ("[laugh]", "[sigh]", "[gasp]") if tag in text)
    assert "five dollars" in normalize_text(Row(**{**row("a").to_json(), "text": "[sigh] I paid $5."}), normalizer)

    class Stripping:
        def normalize(self, text, language):
            return text.replace("[", "").replace("]", "")

    with pytest.raises(ValueError, match="event tags"):
        normalize_text(row("a"), Stripping())
