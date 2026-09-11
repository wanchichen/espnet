"""VoxPopuli ASR dataset module."""

from egs3.voxpopuli.asr.dataset.builder import VoxPopuliBuilder as DatasetBuilder
from egs3.voxpopuli.asr.dataset.dataset import VoxPopuliDataset as Dataset
from egs3.voxpopuli.asr.dataset.dataset import gather_training_text

__all__ = ["Dataset", "DatasetBuilder", "gather_training_text"]
