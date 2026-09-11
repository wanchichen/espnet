"""Inference output formatting for the VoxPopuli ASR recipe.

`conf/inference.yaml` points `output_fn` here. ESPnet3 calls it once per sample
and writes one SCP file per returned key under
`${inference_dir}/<test_name>/`, so the keys below become `hyp.scp` and
`ref.scp` -- the two files `conf/metrics.yaml` scores.
"""


def build_output(data, model_output, idx):
    """Turn one Speech2Text result into the dict ESPnet3 writes to SCP.

    Args:
        data: The raw dataset sample. `VoxPopuliDataset.__getitem__` returns
            only `speech` and `text`; see the note on `utt_id` below.
        model_output: `espnet2.bin.asr_inference.Speech2Text.__call__` output,
            an n-best list of `(text, token, token_int, hypothesis)` tuples, so
            `[0][0]` is the best hypothesis text.
        idx: Index of the sample within its test set.

    Returns:
        dict with `utt_id`, `hyp` and `ref`.

    The VoxPopuli segment id is deliberately not carried in the dataset sample:
    espnet2's CommonPreprocessor is typechecked as returning
    Dict[str, np.ndarray], so a string key aborts collect_stats and training
    (see dataset/dataset.py:__getitem__). The sample index is used instead,
    exactly as egs3/librispeech_100 and egs3/spgispeech do. Scoring is
    unaffected -- `ref` and `hyp` come from one pass over one ordering, so the
    two SCP files line up key for key.
    """
    utt_id = data.get("utt_id", str(idx))
    hyp = model_output[0][0]
    ref = data.get("text", "")
    return {"utt_id": utt_id, "hyp": hyp, "ref": ref}
