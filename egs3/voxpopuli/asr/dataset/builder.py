"""VoxPopuli dataset builder.

VoxPopuli is not downloadable from this recipe. The corpus is ~500 GB of raw
European Parliament recordings before segmentation, and the upstream tooling
(https://github.com/facebookresearch/voxpopuli) already handles the download,
the VAD-based segmentation, and the transcript alignment. Reproducing that here
would duplicate it badly. The builder therefore only locates and validates the
tree that upstream produced:

    python -m voxpopuli.download_audios --root ROOT --subset asr
    python -m voxpopuli.get_asr_data    --root ROOT --lang LANG

which leaves

    ROOT/transcribed_data/LANG/asr_{train,dev,test}.tsv
    ROOT/transcribed_data/LANG/YYYY/<segment_id>.ogg   (16 kHz mono Ogg Vorbis)

There is no build step: `dataset.py` reads those manifests and audio files
directly, so `build()` is the same validation as `prepare_source()`.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Iterable

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults


def _load_config() -> dict:
    """Return the parsed ``dataset/config.yaml``."""
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)


_CONFIG = _load_config()
_CFG = _CONFIG["builder"]
_DATASET_CFG = _CONFIG["dataset"]

DEFAULT_LANG = "en"
SUPPORTED_LANGUAGES = [str(lang) for lang in _DATASET_CFG["supported_languages"]]
TEST_ONLY_LANGUAGES = frozenset(
    str(lang) for lang in _DATASET_CFG["test_only_languages"]
)
SUPPORTED_SPLITS = [str(split) for split in _DATASET_CFG["supported_splits"]]


def check_language(lang: str) -> str:
    """Validate a language code against ``supported_languages``.

    Args:
        lang: VoxPopuli language code, e.g. ``en`` or ``en_accented``.

    Returns:
        The validated language code.

    Raises:
        ValueError: If the code has no transcribed ASR data.
    """
    lang = str(lang)
    if lang not in SUPPORTED_LANGUAGES:
        known = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(
            f"Unknown VoxPopuli ASR language '{lang}'. Expected one of: {known}. "
            "Languages outside this list have unlabelled audio only."
        )
    return lang


def required_splits(lang: str) -> list[str]:
    """Return the split manifests that must exist for ``lang``."""
    if lang in TEST_ONLY_LANGUAGES:
        return ["test"]
    return list(SUPPORTED_SPLITS)


def manifest_path(language_root: Path, split: str) -> Path:
    """Return the manifest path for one split inside a language directory."""
    return language_root / str(_CFG["manifest_template"]).format(split=split)


def _is_language_root(candidate: Path, lang: str) -> bool:
    """Return whether ``candidate`` is a populated VoxPopuli language directory."""
    if not candidate.is_dir():
        return False
    return all(
        manifest_path(candidate, split).is_file() for split in required_splits(lang)
    )


def iter_language_root_candidates(
    recipe_root: Path,
    lang: str,
    source_dir: str | Path | None,
) -> Iterable[Path]:
    """Yield directories that may be ``transcribed_data/<lang>`` for ``lang``.

    Each base directory is tried at three depths, so a user may point the recipe
    at the VoxPopuli ROOT, at its ``transcribed_data`` directory, or straight at
    one language directory. The third form matters in practice: shared corpus
    copies often rename or relocate ``transcribed_data``.

    Precedence is explicit ``source_dir`` first, so a caller that names a corpus
    path is never silently overridden by a recipe-local ``download/`` tree or by
    a stale environment variable.
    """
    bases: list[Path] = []
    if source_dir is not None and str(source_dir):
        bases.append(Path(source_dir))
    bases.append(Path(recipe_root) / str(_CFG["dataset_path"]))
    env_path = os.environ.get(str(_CFG["source_env_var"]))
    if env_path:
        bases.append(Path(env_path))

    transcribed_subdir = str(_CFG["transcribed_subdir"])
    for base in bases:
        yield base / transcribed_subdir / lang
        yield base / lang
        yield base


def resolve_language_root(
    recipe_root: Path,
    lang: str = DEFAULT_LANG,
    source_dir: str | Path | None = None,
) -> Path:
    """Resolve the ``transcribed_data/<lang>`` directory for this recipe.

    Args:
        recipe_root: Recipe root directory.
        lang: VoxPopuli language code.
        source_dir: Optional path to the VoxPopuli ROOT, to its
            ``transcribed_data`` directory, or to the language directory itself.

    Returns:
        The resolved language directory.

    Raises:
        ValueError: If ``lang`` has no transcribed ASR data.
        FileNotFoundError: If no candidate directory holds the required
            manifests.
    """
    lang = check_language(lang)
    recipe_root = Path(recipe_root)
    checked: list[str] = []
    for candidate in iter_language_root_candidates(recipe_root, lang, source_dir):
        checked.append(str(candidate))
        if _is_language_root(candidate, lang):
            return candidate.resolve()

    wanted = ", ".join(
        str(_CFG["manifest_template"]).format(split=split)
        for split in required_splits(lang)
    )
    raise FileNotFoundError(
        f"VoxPopuli '{lang}' transcribed data not found. Each of these was "
        f"checked for {wanted}:\n"
        + "\n".join(f"  - {path}" for path in checked)
        + "\n\nThis recipe does not download VoxPopuli. Follow "
        "https://github.com/facebookresearch/voxpopuli:\n"
        "    python -m voxpopuli.download_audios --root ROOT --subset asr\n"
        f"    python -m voxpopuli.get_asr_data    --root ROOT --lang {lang}\n"
        f"then set {_CFG['source_env_var']}=ROOT, or point the training config's "
        "`dataset_dir` at it."
    )


def missing_manifests(language_root: Path, lang: str) -> list[str]:
    """Return the manifest file names that are missing for ``lang``."""
    return [
        manifest_path(language_root, split).name
        for split in required_splits(lang)
        if not manifest_path(language_root, split).is_file()
    ]


def has_segment_directory(language_root: Path) -> bool:
    """Return whether any ``YYYY`` segment directory exists.

    ``download_audios`` and ``get_asr_data`` are separate upstream steps, and the
    manifests are downloaded by the second one before it segments any audio. A
    language directory can therefore hold complete manifests and no audio at
    all, which is worth catching in `create_dataset` rather than on the first
    training batch.
    """
    return any(
        child.is_dir() and child.name.isdigit() and len(child.name) == 4
        for child in language_root.iterdir()
    )


class VoxPopuliBuilder(DatasetBuilder):
    """Validate that the VoxPopuli transcribed data for one language is present.

    The recipe reads the upstream manifests and Ogg files directly, so there is
    nothing to build. `is_built`/`build` delegate to the source checks.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        lang: str = DEFAULT_LANG,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the language's manifests and audio are available."""
        recipe_root = Path(recipe_dir).resolve()
        try:
            language_root = resolve_language_root(
                recipe_root, lang=lang, source_dir=source_dir
            )
        except (FileNotFoundError, ValueError):
            return False
        return has_segment_directory(language_root)

    def prepare_source(
        self,
        recipe_dir: str | Path,
        lang: str = DEFAULT_LANG,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Validate the VoxPopuli source tree, explaining how to build it.

        Args:
            recipe_dir: Recipe root directory.
            lang: VoxPopuli language code.
            source_dir: Optional path to the corpus, at any of the three depths
                accepted by `resolve_language_root`.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            ValueError: If ``lang`` has no transcribed ASR data.
            FileNotFoundError: If the manifests or the segmented audio are
                missing.
        """
        recipe_root = Path(recipe_dir).resolve()
        language_root = resolve_language_root(
            recipe_root, lang=lang, source_dir=source_dir
        )

        missing = missing_manifests(language_root, lang)
        if missing:
            raise FileNotFoundError(
                f"VoxPopuli '{lang}' is incomplete at {language_root}. "
                "Missing manifests: " + ", ".join(missing)
            )

        if not has_segment_directory(language_root):
            raise FileNotFoundError(
                f"VoxPopuli '{lang}' has manifests but no segmented audio under "
                f"{language_root} (no YYYY/ directory). The manifests are "
                "downloaded before the audio is cut, so an interrupted run "
                "leaves exactly this state. Re-run:\n"
                f"    python -m voxpopuli.get_asr_data --root ROOT --lang {lang}"
            )

    def is_built(
        self,
        recipe_dir: str | Path,
        lang: str = DEFAULT_LANG,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return source readiness because this recipe has no build artifacts."""
        return self.is_source_prepared(
            recipe_dir=recipe_dir,
            lang=lang,
            source_dir=source_dir,
        )

    def build(
        self,
        recipe_dir: str | Path,
        lang: str = DEFAULT_LANG,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """No-op build step: validate the source tree and return."""
        self.prepare_source(
            recipe_dir=recipe_dir,
            lang=lang,
            source_dir=source_dir,
        )
