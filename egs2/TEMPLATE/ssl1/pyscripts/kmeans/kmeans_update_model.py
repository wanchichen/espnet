#!/usr/bin/python

import argparse
import logging
import os
import sys

import numpy as np
import pickle

from kmeans_model import KMeansModel

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
        "--kmeans_init_ckpt", type=str, default=None, help="initial / pretrained checkpoint for kmeans."
    )
    parser.add_argument(
        "in_stats_files", type=str, nargs="+", help="Read specifier for feats. e.g. ark:some.ark"
    )

    return parser


def main(args):
    if isinstance(args.in_stats_files, str):
        in_stats_files = [args.in_stats_files]
    else:
        in_stats_files = args.in_stats_files

    with open(args.kmeans_init_ckpt, "rb") as f:
        ckpt = pickle.load(f)

    stats = np.zeros((ckpt.n_clusters, ckpt.n_features+1), dtype=np.float64)
    for f in in_stats_files:
        in_stats = np.load(f, allow_pickle=True)
        stats += in_stats

    new_centroids = stats[:, :-1] / stats[:, -1:]

    if all([np.allclose(x, y) for x, y in zip(ckpt.centroids, new_centroids)]) :
        return 1
    else:
        kmeans_model = KMeansModel(
            n_clusters = ckpt.n_clusters,
            n_features = ckpt.n_features,
            centroids = new_centroids,
        )
        with open(args.kmeans_init_ckpt, "wb") as f:
            pickle.dump(kmeans_model, f, pickle.HIGHEST_PROTOCOL)
        return 0


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()

    ret = main(args)
    print(ret)