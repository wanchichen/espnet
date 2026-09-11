"""Tests for egs3/voxpopuli/asr/dataset/dataset.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from egs3.voxpopuli.asr.dataset.dataset import (
    VoxPopuliDataset,
    audio_path_for,
    gather_training_text,
)


def test_audio_path_uses_the_year_prefix_of_the_segment_id(tmp_path: Path) -> None:
    utt_id = "20130114-0900-PLENARY-1-en_20130114-17:12:30_0"
    assert audio_path_for(tmp_path, utt_id) == tmp_path / "2013" / f"{utt_id}.ogg"


def test_rows_with_an_empty_target_are_dropped(
    tmp_path: Path, voxpopuli_root: Path
) -> None:
    dataset = VoxPopuliDataset(
        split="train", lang="en", recipe_dir=tmp_path, source_dir=voxpopuli_root
    )
    # Three manifest rows, one with an empty normalized_text.
    assert len(dataset) == 2
    assert [dataset[i]["text"] for i in range(len(dataset))] == [
        "madam president",
        "he said no",
    ]


def test_getitem_returns_only_speech_and_text(
    tmp_path: Path, voxpopuli_root: Path
) -> None:
    dataset = VoxPopuliDataset(
        split="dev", lang="en", recipe_dir=tmp_path, source_dir=voxpopuli_root
    )
    sample = dataset[0]
    # A str under any other key would fail espnet2's typechecked preprocessor
    # and then the collate function; see dataset.py:__getitem__.
    assert sorted(sample) == ["speech", "text"]
    assert isinstance(sample["speech"], np.ndarray)
    assert sample["speech"].dtype == np.float32
    assert sample["text"] == "thank you"


def test_raw_text_keeps_casing_and_punctuation(
    tmp_path: Path, voxpopuli_root: Path
) -> None:
    dataset = VoxPopuliDataset(
        split="train",
        lang="en",
        recipe_dir=tmp_path,
        source_dir=voxpopuli_root,
        text_field="raw_text",
    )
    assert dataset[0]["text"] == "Madam President."
    # The manifest is unquoted TSV, so a double quote is literal text.
    assert dataset[1]["text"] == 'He said "no".'


def test_unknown_split_language_and_text_field_are_rejected(
    tmp_path: Path, voxpopuli_root: Path
) -> None:
    common = dict(lang="en", recipe_dir=tmp_path, source_dir=voxpopuli_root)
    with pytest.raises(ValueError, match="Unknown split"):
        VoxPopuliDataset(split="valid", **common)
    with pytest.raises(ValueError, match="Unknown text_field"):
        VoxPopuliDataset(split="dev", text_field="transcript", **common)
    with pytest.raises(ValueError, match="Unknown VoxPopuli ASR language"):
        VoxPopuliDataset(
            split="dev", lang="pt", recipe_dir=tmp_path, source_dir=voxpopuli_root
        )


def test_accented_english_has_no_train_split(
    tmp_path: Path, voxpopuli_root: Path
) -> None:
    with pytest.raises(ValueError, match="has no 'train' split"):
        VoxPopuliDataset(
            split="train",
            lang="en_accented",
            recipe_dir=tmp_path,
            source_dir=voxpopuli_root,
        )


def test_a_manifest_without_its_audio_names_the_missing_file(
    tmp_path: Path, voxpopuli_root: Path
) -> None:
    language_root = voxpopuli_root / "transcribed_data" / "en"
    first = sorted((language_root / "2013").iterdir())[0]
    first.unlink()
    with pytest.raises(FileNotFoundError, match=first.name):
        VoxPopuliDataset(
            split="train", lang="en", recipe_dir=tmp_path, source_dir=voxpopuli_root
        )


def test_gather_training_text_reads_the_manifest_without_audio(
    tmp_path: Path, manifests_only_root: Path
) -> None:
    texts = gather_training_text(
        recipe_dir=tmp_path, lang="en", source_dir=manifests_only_root
    )
    assert texts == ["madam president", "he said no"]
