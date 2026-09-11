"""VoxPopuli ASR dataset backed by the upstream transcribed data.

`python -m voxpopuli.get_asr_data --root ROOT --lang LANG` writes, for one
language:

    ROOT/transcribed_data/LANG/asr_{train,dev,test}.tsv
    ROOT/transcribed_data/LANG/YYYY/<segment_id>.ogg

The manifests are tab-separated with the header

    id  raw_text  normalized_text  speaker_id  split  gender
    is_gold_transcript  accent

and one row per segment. The ``id`` doubles as the audio path: it starts with
the four-digit year of the parliamentary session, so segment ``20130114-...``
lives in ``2013/20130114-....ogg``. This module reads that layout directly --
there is no intermediate manifest, no Kaldi data directory, and no copy of the
audio.
"""

from __future__ import annotations

import csv
import functools
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.voxpopuli.asr.dataset.builder import (
    DEFAULT_LANG,
    VoxPopuliBuilder,
    check_language,
    manifest_path,
    required_splits,
    resolve_language_root,
)
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}
DEFAULT_TEXT_FIELD = str(_DATASET_CFG["text_field"])
_TEXT_FIELDS = ("normalized_text", "raw_text")

AUDIO_SUFFIX = ".ogg"
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class VoxPopuliExample:
    """One manifest row, resolved to an audio path."""

    utt_id: str
    audio_path: Path
    text: str


def audio_path_for(language_root: Path, utt_id: str) -> Path:
    """Return the segment path for ``utt_id`` under a language directory.

    The year directory is the first four characters of the segment id, which is
    how ``voxpopuli/get_asr_data.py`` lays the files out.
    """
    return language_root / utt_id[:4] / f"{utt_id}{AUDIO_SUFFIX}"


@functools.lru_cache(maxsize=16)
def _load_manifest(
    language_root: Path,
    split: str,
    text_field: str,
) -> tuple[VoxPopuliExample, ...]:
    """Parse one ``asr_<split>.tsv`` into an index, in manifest order.

    Rows whose chosen text column is empty are dropped: VoxPopuli ships a small
    number of segments with an empty transcript (16 of 182,482 in the English
    training set), and an empty target crashes CTC with an input-shorter-than-
    target error rather than merely wasting a step.
    """
    tsv_path = manifest_path(language_root, split)
    examples: list[VoxPopuliExample] = []
    empty_text = 0

    with tsv_path.open("r", encoding="utf-8", newline="") as stream:
        # QUOTE_NONE: get_asr_data.py writes "\t".join(cols) with no quoting, so
        # a transcript containing a double quote is literal text, not a quoted
        # field. csv's default QUOTE_MINIMAL would swallow those characters and
        # could merge columns.
        reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty VoxPopuli manifest: {tsv_path}")
        if text_field not in reader.fieldnames:
            raise RuntimeError(
                f"Column '{text_field}' is not in {tsv_path}. "
                f"Available columns: {', '.join(reader.fieldnames)}"
            )
        for row in reader:
            utt_id = (row.get("id") or "").strip()
            text = (row.get(text_field) or "").strip()
            if not utt_id:
                continue
            if not text:
                empty_text += 1
                continue
            examples.append(
                VoxPopuliExample(
                    utt_id=utt_id,
                    audio_path=audio_path_for(language_root, utt_id),
                    text=text,
                )
            )

    if not examples:
        raise RuntimeError(
            f"No usable rows in {tsv_path} "
            f"({empty_text} row(s) had an empty '{text_field}'). "
            "Check that the manifest is the one written by "
            "`python -m voxpopuli.get_asr_data`."
        )

    return tuple(examples)


def _check_first_segment(examples: tuple[VoxPopuliExample, ...], split: str) -> None:
    """Check that the first segment of a split exists on disk.

    One stat, not one per utterance. Indexing the English training set means
    182k rows, and stat()ing each of them costs minutes on a network filesystem
    for a check that only ever fires on a half-segmented corpus. A file missing
    later surfaces as a soundfile error naming the path.
    """
    first = examples[0]
    if not first.audio_path.is_file():
        language_root = first.audio_path.parent.parent
        raise FileNotFoundError(
            f"Manifest asr_{split}.tsv lists segment '{first.utt_id}' but "
            f"{first.audio_path} does not exist. Run\n"
            "    python -m voxpopuli.get_asr_data --root ROOT --lang "
            f"{language_root.name}\n"
            "to segment the raw audio for this language."
        )


class VoxPopuliDataset(TorchDataset):
    """Torch dataset over one VoxPopuli language and split.

    Args:
        split: ``train``, ``dev`` or ``test``. ``en_accented`` has ``test`` only.
        lang: VoxPopuli language code (default ``en``).
        recipe_dir: Optional recipe root. Defaults to the recipe directory this
            module lives in.
        source_dir: Optional path to the VoxPopuli ROOT, its
            ``transcribed_data`` directory, or the language directory itself.
            When omitted, ``<recipe_dir>/download/voxpopuli`` and then the
            ``VOXPOPULI`` environment variable are tried.
        text_field: Manifest column used as the target, ``normalized_text``
            (default) or ``raw_text``.

    Raises:
        ValueError: If ``split``, ``lang`` or ``text_field`` is unknown, or if
            the split does not exist for the language.
        FileNotFoundError: If the corpus or the segmented audio is missing.
        RuntimeError: If the manifest yields no usable rows.

    Examples:
        >>> dataset = VoxPopuliDataset(split="dev", lang="en")
        >>> sample = dataset[0]
        >>> sorted(sample.keys())
        ['speech', 'text']
    """

    def __init__(
        self,
        split: str,
        lang: str = DEFAULT_LANG,
        recipe_dir: str | Path | None = None,
        source_dir: str | Path | None = None,
        text_field: str = DEFAULT_TEXT_FIELD,
    ) -> None:
        self.split = str(split)
        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        self.lang = check_language(lang)
        if self.split not in required_splits(self.lang):
            available = ", ".join(required_splits(self.lang))
            raise ValueError(
                f"VoxPopuli '{self.lang}' has no '{self.split}' split. "
                f"Available: {available}."
            )

        self.text_field = str(text_field)
        if self.text_field not in _TEXT_FIELDS:
            known = ", ".join(_TEXT_FIELDS)
            raise ValueError(
                f"Unknown text_field '{self.text_field}'. Expected one of: {known}"
            )

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )

        builder = VoxPopuliBuilder()
        if not builder.is_source_prepared(
            recipe_dir=recipe_root,
            lang=self.lang,
            source_dir=source_dir,
        ):
            builder.prepare_source(
                recipe_dir=recipe_root,
                lang=self.lang,
                source_dir=source_dir,
            )

        self.language_root = resolve_language_root(
            recipe_root, lang=self.lang, source_dir=source_dir
        )
        self._examples = _load_manifest(
            self.language_root, self.split, self.text_field
        )
        _check_first_segment(self._examples, self.split)

    def __len__(self) -> int:
        """Return the number of usable segments in this split."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return ``{speech, text}`` for one segment.

        No ``utt_id`` key: espnet2's ``CommonPreprocessor`` is typechecked as
        returning ``Dict[str, np.ndarray]`` and passes keys it does not know
        through unchanged, so a string here aborts ``collect_stats`` and
        training. ESPnet3 identifies samples by index instead, and
        ``src/inference.py`` falls back to that index for the SCP key.
        """
        example = self._examples[int(idx)]
        array, _sr = sf.read(str(example.audio_path))
        return {
            "speech": np.asarray(array, dtype=np.float32),
            "text": example.text,
        }


def gather_training_text(
    recipe_dir: str | Path,
    lang: str = DEFAULT_LANG,
    source_dir: str | Path | None = None,
    split: str = "train",
    text_field: str = DEFAULT_TEXT_FIELD,
    **_kwargs: Any,
) -> list[str]:
    """Collect transcripts for tokenizer training.

    Reads the manifest only, so it does not touch a single audio file. The
    tokenizer can therefore be trained while the audio is still being segmented.

    Args:
        recipe_dir: Recipe root directory.
        lang: VoxPopuli language code.
        source_dir: Optional corpus path override.
        split: Manifest split to read text from.
        text_field: Manifest column to read.
        **_kwargs: Unused extra options for API compatibility.

    Returns:
        One transcript string per usable segment.
    """
    recipe_root = Path(recipe_dir).resolve()
    language_root = resolve_language_root(
        recipe_root, lang=lang, source_dir=source_dir
    )
    return [
        example.text
        for example in _load_manifest(language_root, str(split), str(text_field))
    ]
