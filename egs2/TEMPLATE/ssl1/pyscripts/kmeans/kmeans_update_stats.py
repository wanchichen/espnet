#!/usr/bin/python

import argparse
from contextlib import nullcontext
import logging
import os
import sys
import time

import numpy as np
import pickle

from kmeans_model import KMeansModel
from espnet.utils.cli_readers import file_reader_helper
from espnet2.utils.types import str2bool

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("kmeans update labels")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_filetype",
        type=str,
        default="mat",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "--kmeans_init_ckpt", type=str, default=None, help="initial / pretrained checkpoint for kmeans."
    )
    parser.add_argument("--n_clusters", type=int, default=100, help="number of clusters in kmeans model.")
    parser.add_argument("--n_threads", type=int, default=1, help="number of threads in kmeans model.")
    parser.add_argument("--batch_frames", type=int, default=24000, help="batch size in training kmeans model.")
    parser.add_argument("--initialize_ckpt", type=str2bool, default=False)

    parser.add_argument(
        "--dump_label",
        type=str2bool,
        default=False,
        help="whether to dump the predicted labels in E-step.",
    )

    parser.add_argument(
        "rspecifier", type=str, help="Read specifier for feats. e.g. ark:some.ark"
    )
    parser.add_argument(
        "output_label_file", type=str, help="output label file."
    )
    parser.add_argument(
        "output_stats_file", type=str, help="output stats file."
    )

    return parser


def pad_list(mats, lens, pad_value=0):
    n_batch = len(mats)
    max_len = max(lens)
    pad = np.zeros((n_batch, max_len, *mats[0].shape[1:])) + pad_value

    for i in range(n_batch):
        pad[i, : lens[i]] = mats[i]

    return pad


def main(args):
    start_time = time.time()
    # Read in sample data
    for utt, mat in file_reader_helper(args.rspecifier, args.in_filetype):
        sample_data = mat
        break

    # initialize kmeans model
    kmeans_model = KMeansModel(
        n_clusters = args.n_clusters,
        n_features = sample_data.shape[1],
        n_threads = args.n_threads,
    )

    if args.initialize_ckpt:
        length = 0
        init_data = []

        # Accum initial data for initialization.
        # for utt, mat in file_reader_helper(args.rspecifier, args.in_filetype):
        #     init_data.append(mat)
        #     length += mat.shape[0]
        #     if length > 2000000:  # ~ 10 hr hubert features
        #         break

        # X = np.concatenate(init_data, axis=0)
        # kmeans_model.centroids, _ = initialize_kmeans(
        #     X,
        #     args.n_clusters,
        #     random_state=np.random.RandomState(0),
        #     x_squared_norms=np.einsum("ij,ij->i", X, X),
        # )
        with open(args.kmeans_init_ckpt, "wb") as f:
            pickle.dump(kmeans_model, f, pickle.HIGHEST_PROTOCOL)
        return
    else:
        kmeans_model.load_ckpt(args.kmeans_init_ckpt)

    # Dump labels and stats in the file
    out_root = os.path.dirname(args.output_label_file)
    os.makedirs(out_root, exist_ok=True)
    utt_ids, mats, lens = [], [], []
    stats = np.zeros((args.n_clusters, sample_data.shape[1]), dtype=np.float64)
    counts = np.zeros(args.n_clusters, dtype=np.int64)

    frames_cnt = 0
    predict_time = 0

    with open(args.output_label_file, "w") if args.dump_label else nullcontext() as f:
        for utt, mat in file_reader_helper(args.rspecifier, args.in_filetype):
            utt_ids.append(utt)
            mats.append(mat)
            lens.append(mat.shape[0])
            frames_cnt += lens[-1]

            if frames_cnt < args.batch_frames:
                continue
            else:
                # (seq_len, dim)
                mats_pad = np.concatenate(mats, axis=0)  # (total_seq_len, dim)
                s_t = time.time()
                labels = kmeans_model.predict(mats_pad)
                e_t = time.time()
                predict_time += e_t - s_t

                accum_lens = np.cumsum(np.array(lens))
                labels = np.split(labels, accum_lens[:-1])
                for i, (utt_id, label) in enumerate(zip(utt_ids, labels)):
                    for j in range(lens[i]):
                        stats[label[j]] += mats[i][j]
                    uniq, cnt = np.unique(np.array(label[:lens[i]]), return_counts=True)
                    for u, c in zip(uniq, cnt):
                        counts[u] += c

                utt_ids, mats, lens = [], [], []

    np.save(
        args.output_stats_file,
        np.concatenate([stats, counts[:, None]], axis=1, dtype=np.float64()),
        allow_pickle=True,
    )

    end_time = time.time()
    logging.info(f"N_threads {args.n_threads} Batch_frames {args.batch_frames} Time elapsed: {end_time - start_time} Predict Time elapsed: {predict_time}")


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)