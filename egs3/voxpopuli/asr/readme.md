# VoxPopuli ASR recipe

[VoxPopuli](https://github.com/facebookresearch/voxpopuli) (Wang et al., ACL
2021) is 2009-2020 European Parliament plenary audio. This recipe uses its
**transcribed** subset: 1,791 h of aligned speech over 16 languages, plus a
29 h accented-English test set. Segments are 16 kHz mono Ogg Vorbis, cut by VAD
and aligned to the official transcripts.

## Getting the data

**This recipe does not download VoxPopuli.** The raw ASR audio is several
hundred GB before segmentation, and the upstream repository already owns the
download, VAD segmentation and transcript alignment. Follow its instructions:

```bash
git clone https://github.com/facebookresearch/voxpopuli.git
cd voxpopuli && pip install -r requirements.txt

# raw session audio, shared by every language (~500 GB)
python -m voxpopuli.download_audios --root ROOT --subset asr

# segment and align one language
python -m voxpopuli.get_asr_data --root ROOT --lang en
```

That produces the only layout this recipe reads:

```
ROOT/transcribed_data/en/asr_train.tsv      id, raw_text, normalized_text,
ROOT/transcribed_data/en/asr_dev.tsv          speaker_id, split, gender,
ROOT/transcribed_data/en/asr_test.tsv         is_gold_transcript, accent
ROOT/transcribed_data/en/2013/<segment_id>.ogg
```

Then point the recipe at it, in whichever of these three ways suits your setup
(explicit `dataset_dir` wins over the environment variable):

```bash
export VOXPOPULI=/path/to/ROOT          # or
# conf/tuning/train_asr_e_branchformer.yaml:  dataset_dir: /path/to/ROOT
# or place/symlink the corpus at            <recipe_dir>/download/voxpopuli
```

`dataset_dir` and `$VOXPOPULI` are accepted at any of three depths: the ROOT
itself, its `transcribed_data` directory, or a single
`transcribed_data/<lang>` directory. The third form matters for shared corpus
copies, where `transcribed_data` is often renamed or mounted elsewhere.

`create_dataset` only validates this tree -- it checks that the manifests for
the language exist and that at least one `YYYY/` segment directory is present.
The manifests are downloaded before the audio is cut, so a half-finished
`get_asr_data` run leaves manifests with no audio; that state is reported there
rather than on the first training batch.

## Quick start

```bash
# 1) Validate the corpus for the configured language
python run.py --stages create_dataset \
    --training_config conf/tuning/train_asr_e_branchformer.yaml

# 2) Train the BPE-5000 tokenizer (reads manifests only, no audio)
python run.py --stages train_tokenizer \
    --training_config conf/tuning/train_asr_e_branchformer.yaml

# 3) Collect feature statistics (global_mvn + batch shapes)
python run.py --stages collect_stats \
    --training_config conf/tuning/train_asr_e_branchformer.yaml

# 4) Train
python run.py --stages train \
    --training_config conf/tuning/train_asr_e_branchformer.yaml

# 5) Decode and score dev + test
python run.py --stages infer measure \
    --training_config conf/tuning/train_asr_e_branchformer.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

## Choosing a language

Set `lang` in the training config (and in `conf/inference.yaml`, which must
match). `data_dir`, `stats_dir` and `exp_dir` all include `${lang}`, so runs for
different languages never share a tokenizer, statistics or experiment
directory.

| lang | language | transcribed h | | lang | language | transcribed h |
| --- | --- | ---: | --- | --- | --- | ---: |
| `en` | English | 543 | | `cs` | Czech | 62 |
| `de` | German | 282 | | `nl` | Dutch | 53 |
| `fr` | French | 211 | | `hr` | Croatian | 43 |
| `es` | Spanish | 166 | | `sk` | Slovak | 35 |
| `pl` | Polish | 111 | | `fi` | Finnish | 27 |
| `it` | Italian | 91 | | `sl` | Slovene | 10 |
| `ro` | Romanian | 89 | | `et` | Estonian | 3 |
| `hu` | Hungarian | 63 | | `lt` | Lithuanian | 2 |

`en_accented` (29 h, 15 L2 accents) is a **test set only** and has no `train` or
`dev` manifest. To score a model on it, set `lang: en_accented` in
`conf/inference.yaml` and delete the `dev` entry there.

The shipped hyperparameters are sized for `en`. The small languages need a
smaller vocabulary -- SentencePiece will refuse 5,000 merges on 10 K tokens of
Lithuanian -- and many more epochs.

## Splits and transcripts

The train/dev/test split is VoxPopuli's own, read from the `split` column of the
upstream manifests; the recipe does no resplitting. For English that is
182,466 / 1,753 / 1,842 usable segments.

The target is the `normalized_text` column: lowercased, punctuation stripped,
numbers spelled out. This is what VoxPopuli's own ASR baselines are scored on,
so `conf/metrics.yaml` applies no text cleaner. Set `text_field: raw_text` in
the training config to train a cased and punctuated model instead -- the WER is
then not comparable to the paper, and you should add a cleaner to the metrics
config.

Rows whose chosen text column is empty are dropped (16 of 182,482 in English
train). An empty target is not a wasted step but a CTC crash, since the input
cannot be shorter than the target.

## Results

None yet. The configuration has been validated end to end -- `create_dataset`,
`train_tokenizer`, `collect_stats`, `train`, `infer` and `measure` all run on
this recipe -- but no full-length training run has been completed, so there is
no WER to publish and no packed model. `conf/publication.yaml` is included for
when there is.

## Notes

- **No language model.** ESPnet3 has no LM stage, so decoding is CTC/attention
  joint decoding without shallow fusion.
- **Model.** E-Branchformer (12 blocks, 256-d) with a 6-block Transformer
  decoder and `ctc_weight: 0.3`, the same architecture as
  `egs3/librispeech_100/asr`. Epochs, vocabulary and batching are sized for
  VoxPopuli in the training config, with the arithmetic written out there.
- **Segment lengths.** VoxPopuli segments are VAD-derived and vary far more than
  LibriSpeech utterances, so SpecAugment masks time by ratio rather than by a
  fixed frame count.
