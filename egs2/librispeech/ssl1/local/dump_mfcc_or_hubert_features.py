# The sklearn_km.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/learn_kmeans.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

import argparse
import logging
import os
from multiprocessing.sharedctypes import Value
from random import sample

import numpy as np
import tqdm
from hubert_feature_loader import (
    ESPnetHubertFeatureReader,
    HubertFeatureReader,
    MfccFeatureReader,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("sklearn_kmeans")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="folder contains wav.scp for training.",
    )
    parser.add_argument(
        "--feat_dir", type=str, required=True, help="folder to save extracted features."
    )
    parser.add_argument("--split", type=str, default=None, required=True)
    parser.add_argument(
        "--feature_type", type=str, default="mfcc", choices=["mfcc", "hubert"]
    )
    parser.add_argument("--hubert-model-url", type=str, default=None)
    parser.add_argument("--hubert-model-path", type=str, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--nshard", type=int, default=None, required=True)
    parser.add_argument("--rank", type=str, default=None, required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--hubert_type",
        type=str,
        default="espnet",
        choices=["espnet", "fairseq"],
        help="Whether the HuBERT encoder implementation is based on espnet or fairseq.",
    )

    return parser


def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} " f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]

        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)

    return iterate, len(lines)


def dump_feature(reader, generator, num, split, nshard, rank, feat_dir):
    iterator = generator()

    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for path, nsample in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path, nsample)
            feat_f.append(feat.cpu().numpy())
            leng_f.write(f"{len(feat)}\n")
    logger.info("finished successfully")


def main(args):
    np.random.seed(args.seed)
    logging.info("Loading Features")
    if args.feature_type == "mfcc":
        reader = MfccFeatureReader(sample_rate=args.sample_rate)
    elif args.feature_type == "hubert":
        assert 0 < args.layer < 24
        if args.hubert_type == "fairseq":
            logging.warning(
                "Fairseq based HuBERT is deprecated. Please use the torchaudio one."
            )
            reader = HubertFeatureReader(
                hubert_url=args.hubert_model_url,
                hubert_dir_path=args.hubert_model_path,
                layer=args.layer,
                sample_rate=args.sample_rate,
                max_chunk=args.max_chunk,
            )
        elif args.hubert_type == "espnet":
            reader = ESPnetHubertFeatureReader(
                hubert_model_path=args.hubert_model_path,
                layer=args.layer,
                sample_rate=args.sample_rate,
                max_chunk=args.max_chunk,
            )
        else:
            raise ValueError(f"Unknown hubert type {args.hubert_type}")
    else:
        raise ValueError(f"Unknown feature type {args.feature_type}.")

    generator, num = get_path_iterator(
        f"{args.data_dir}/{args.split}.tsv", args.nshard, args.rank
    )
    dump_feature(
        reader, generator, num, args.split, args.nshard, args.rank, args.feat_dir
    )


if __name__ == "__main__":

    try:
        from npy_append_array import NpyAppendArray
    except Exception as e:
        print("Error: npy_append_array is not properly installed.")
        print("Please run: . ./path.sh && python -m pip install npy_append_array")
        raise e

    parser = get_parser()
    args = parser.parse_args()
    args.rank = eval(args.rank)

    logging.info(str(args))

    main(args)
