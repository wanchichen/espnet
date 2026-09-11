"""Tests for egs3/voxpopuli/asr/dataset/builder.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from egs3.voxpopuli.asr.dataset.builder import (
    VoxPopuliBuilder,
    check_language,
    has_segment_directory,
    missing_manifests,
    required_splits,
    resolve_language_root,
)


def test_resolve_language_root_accepts_root_transcribed_and_language_dir(
    tmp_path: Path, voxpopuli_root: Path
) -> None:
    expected = (voxpopuli_root / "transcribed_data" / "en").resolve()
    for source_dir in (
        voxpopuli_root,
        voxpopuli_root / "transcribed_data",
        voxpopuli_root / "transcribed_data" / "en",
    ):
        assert resolve_language_root(tmp_path, "en", source_dir) == expected


def test_resolve_language_root_falls_back_to_the_environment_variable(
    tmp_path: Path, voxpopuli_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOXPOPULI", str(voxpopuli_root))
    assert (
        resolve_language_root(tmp_path, "en", None)
        == (voxpopuli_root / "transcribed_data" / "en").resolve()
    )


def test_explicit_source_dir_wins_over_the_environment_variable(
    tmp_path: Path, voxpopuli_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOXPOPULI", str(tmp_path / "does_not_exist"))
    assert (
        resolve_language_root(tmp_path, "en", voxpopuli_root)
        == (voxpopuli_root / "transcribed_data" / "en").resolve()
    )


def test_resolve_language_root_reports_every_checked_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_language_root(tmp_path, "en", tmp_path / "nowhere")
    message = str(excinfo.value)
    assert "asr_train.tsv" in message
    assert str(tmp_path / "nowhere") in message
    # The message has to say how to obtain the corpus, since nothing downloads it.
    assert "voxpopuli.get_asr_data" in message


def test_check_language_rejects_a_language_without_transcripts() -> None:
    assert check_language("en") == "en"
    with pytest.raises(ValueError, match="Unknown VoxPopuli ASR language"):
        check_language("pt")


def test_required_splits_is_test_only_for_accented_english() -> None:
    assert required_splits("en") == ["train", "dev", "test"]
    assert required_splits("en_accented") == ["test"]


def test_missing_manifests_lists_only_absent_files(voxpopuli_root: Path) -> None:
    language_root = voxpopuli_root / "transcribed_data" / "en"
    assert missing_manifests(language_root, "en") == []
    (language_root / "asr_dev.tsv").unlink()
    assert missing_manifests(language_root, "en") == ["asr_dev.tsv"]


def test_has_segment_directory_ignores_non_year_directories(
    voxpopuli_root: Path, manifests_only_root: Path
) -> None:
    assert has_segment_directory(voxpopuli_root / "transcribed_data" / "en")
    language_root = manifests_only_root / "transcribed_data" / "en"
    (language_root / "notes").mkdir()
    assert not has_segment_directory(language_root)


def test_builder_reports_ready_only_when_audio_is_present(
    tmp_path: Path, voxpopuli_root: Path, manifests_only_root: Path
) -> None:
    builder = VoxPopuliBuilder()
    assert builder.is_source_prepared(
        recipe_dir=tmp_path, lang="en", source_dir=voxpopuli_root
    )
    assert not builder.is_source_prepared(
        recipe_dir=tmp_path, lang="en", source_dir=manifests_only_root
    )


def test_builder_explains_manifests_without_segmented_audio(
    tmp_path: Path, manifests_only_root: Path
) -> None:
    builder = VoxPopuliBuilder()
    with pytest.raises(FileNotFoundError, match="no segmented audio"):
        builder.prepare_source(
            recipe_dir=tmp_path, lang="en", source_dir=manifests_only_root
        )


def test_builder_build_is_the_source_check(
    tmp_path: Path, voxpopuli_root: Path
) -> None:
    builder = VoxPopuliBuilder()
    assert builder.is_built(recipe_dir=tmp_path, lang="en", source_dir=voxpopuli_root)
    builder.build(recipe_dir=tmp_path, lang="en", source_dir=voxpopuli_root)
