"""Fixtures building a miniature VoxPopuli tree, so no corpus is needed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SAMPLE_RATE = 16000

HEADER = [
    "id",
    "raw_text",
    "normalized_text",
    "speaker_id",
    "split",
    "gender",
    "is_gold_transcript",
    "accent",
]

# (segment id, raw_text, normalized_text). The third training row has an empty
# normalized_text, exactly as a handful of real VoxPopuli rows do.
ROWS = {
    "train": [
        (
            "20130114-0900-PLENARY-1-en_20130114-17:12:30_0",
            "Madam President.",
            "madam president",
        ),
        (
            "20140114-0900-PLENARY-2-en_20140114-17:12:30_1",
            'He said "no".',
            "he said no",
        ),
        ("20150114-0900-PLENARY-3-en_20150114-17:12:30_2", "Untranscribed.", ""),
    ],
    "dev": [
        ("20160114-0900-PLENARY-4-en_20160114-17:12:30_0", "Thank you.", "thank you"),
    ],
    "test": [
        ("20170114-0900-PLENARY-5-en_20170114-17:12:30_0", "Next item.", "next item"),
    ],
}


def _write_manifest(language_root: Path, split: str) -> None:
    lines = ["\t".join(HEADER)]
    for utt_id, raw_text, normalized_text in ROWS[split]:
        lines.append(
            "\t".join(
                [
                    utt_id,
                    raw_text,
                    normalized_text,
                    "spk",
                    split,
                    "female",
                    "False",
                    "None",
                ]
            )
        )
    (language_root / f"asr_{split}.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


@pytest.fixture()
def voxpopuli_root(tmp_path: Path) -> Path:
    """Return a ROOT holding one language with manifests and segment audio."""
    root = tmp_path / "corpus"
    language_root = root / "transcribed_data" / "en"
    language_root.mkdir(parents=True)

    for split, rows in ROWS.items():
        _write_manifest(language_root, split)
        for index, (utt_id, _raw, _norm) in enumerate(rows):
            year_dir = language_root / utt_id[:4]
            year_dir.mkdir(exist_ok=True)
            duration = 0.1 * (index + 1)
            samples = np.zeros(int(SAMPLE_RATE * duration), dtype="float32")
            sf.write(
                str(year_dir / f"{utt_id}.ogg"),
                samples,
                SAMPLE_RATE,
                format="OGG",
                subtype="VORBIS",
            )

    return root


@pytest.fixture()
def manifests_only_root(tmp_path: Path) -> Path:
    """Return a ROOT whose manifests exist but whose audio was never cut."""
    root = tmp_path / "manifests_only"
    language_root = root / "transcribed_data" / "en"
    language_root.mkdir(parents=True)
    for split in ROWS:
        _write_manifest(language_root, split)
    return root
