"""Training manifest: one JSON object per line.

Required: ``id`` (unique, filesystem-safe), ``audio`` (path; relative paths resolve against the manifest's folder),
``text`` (what the model reads, event tags included), ``language`` (a server code of zonos3.languages, e.g.
``en_us``), ``event`` (evaluation label such as ``speech`` or ``laugh``; never fed to the model), ``split``, ``license``
and ``provenance``.

Optional: ``speaker`` and ``source`` (group ids; a source is one original recording), ``ref_audio`` (a separate clip of
the same speaker for speaker conditioning) and ``self_reference`` (true marks a diagnostic row that is allowed to use
its own audio as ``ref_audio``).

Splits: ``train`` and ``heldout`` never share a speaker or a source. ``overfit`` marks a plumbing fixture that is
trained and evaluated on the same rows; it says nothing about generalization.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from zonos3 import languages

SPLITS = ("train", "heldout", "overfit")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REQUIRED = ("id", "audio", "text", "language", "event", "split", "license", "provenance")


@dataclass(frozen=True)
class Row:
    id: str
    audio: str
    text: str
    language: str
    event: str
    split: str
    license: str
    provenance: str
    speaker: str | None = None
    source: str | None = None
    ref_audio: str | None = None
    self_reference: bool = False

    def groups(self) -> list[str]:
        """The ids that must stay on one side of the train / heldout boundary."""
        named = [f"{kind}:{value}" for kind, value in (("speaker", self.speaker), ("source", self.source)) if value]
        return named or [f"row:{self.id}"]

    def to_json(self) -> dict:
        return asdict(self)


def load_manifest(path: str | Path) -> list[Row]:
    path = Path(path)
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        doc = json.loads(line)
        missing = [key for key in _REQUIRED if doc.get(key) in (None, "")]
        if missing:
            raise ValueError(f"{path}:{number}: missing {missing}")
        for key in ("audio", "ref_audio"):
            if doc.get(key):
                doc[key] = str((path.parent / doc[key]).resolve())
        rows.append(Row(**doc))
    validate(rows)
    return rows


def validate(rows: list[Row]) -> None:
    seen: set[str] = set()
    side: dict[str, str] = {}
    for row in rows:
        if not _ID.fullmatch(row.id) or row.id in seen:
            raise ValueError(f"id {row.id!r} is not unique and filesystem-safe")
        seen.add(row.id)
        if row.split not in SPLITS:
            raise ValueError(f"{row.id}: split {row.split!r} is not one of {SPLITS}")
        if row.language not in languages.BY_CODE:
            raise ValueError(f"{row.id}: language {row.language!r} is not a server language code")
        if row.ref_audio == row.audio and not row.self_reference:
            raise ValueError(f"{row.id}: ref_audio is the target clip; set self_reference for a diagnostic row")
        if row.split == "overfit":
            continue
        for group in row.groups():
            if side.setdefault(group, row.split) != row.split:
                raise ValueError(f"{group} appears in both train and heldout")
    targets = {row.audio: row.split for row in rows if row.split != "overfit"}
    for row in rows:
        other = targets.get(row.ref_audio)
        if row.split != "overfit" and other not in (None, row.split):
            raise ValueError(f"{row.id}: its ref_audio is a {other} target")


def assign_splits(rows: list[Row], heldout_fraction: float, seed: int = 0) -> list[Row]:
    """``rows`` with ``train`` / ``heldout`` reassigned by group: rows linked through a speaker or a source land on the
    same side, and the groups are ordered by a hash of ``seed`` and their smallest id. ``overfit`` rows are kept."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = x = parent[parent[x]]
        return x

    free = [row for row in rows if row.split != "overfit"]
    for row in free:
        first, *rest = row.groups()
        for other in rest:
            parent[find(other)] = find(first)
    components: dict[str, list[str]] = {}
    for row in free:
        components.setdefault(find(row.groups()[0]), []).append(row.id)
    ordered = sorted(components, key=lambda root: hashlib.sha256(f"{seed}:{min(components[root])}".encode()).digest())
    heldout = set(ordered[: round(heldout_fraction * len(ordered))])
    out = []
    for row in rows:
        if row.split != "overfit":
            split = "heldout" if find(row.groups()[0]) in heldout else "train"
            row = Row(**{**row.to_json(), "split": split})
        out.append(row)
    validate(out)
    return out
