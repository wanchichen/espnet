# Modified from ESPnet's HuBERT SSL k-means, by Xuankai Chang.

import argparse

import numpy as np

from feature_reader import (
    MfccFeatureReader,
    ESPnetModelFeatureReader
)

from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import is_scipy_wav_style
from sklearn.cluster import KMeans
import skfuzzy as fuzz

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_clusters", default=32, type=int)
    parser.add_argument(
        "--in_filetype",
        type=str,
        default="sound",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "rspecifier", type=str, help="Read specifier for feats. e.g. ark:some.ark"
    )

    return parser

def learn_kmeans(
    reader, in_filetype, rspecifier, max_len = 3200, num_clusters=2):

    feats = []
    for utt, mat in file_reader_helper(rspecifier, in_filetype):
        if is_scipy_wav_style(mat):
            # If data is sound file, then got as Tuple[int, ndarray]
            rate, mat = mat
            mat = mat.astype(np.float64, order="C") / 32768.0
        nsample = len(mat)
        feat = reader.get_feats(mat, nsample).numpy()
        print(feat.shape, flush=True)
        pad_width = int(max_len - feat.shape[0])

        feat = np.pad(feat, ((0, pad_width), (0,0)), 'constant', constant_values=((0,), (0,)))
        feats.append(feat)

    feats = np.stack(feats, axis=0) # n x seq x dim
    feats = feats.reshape((feats.shape[0], -1)) # n x (seq x dim)

    #kmeans = KMeans(n_clusters=2, random_state=0, n_init=1).fit(feats)
    p=0
    fpc = 0
    kmeans, _, _, _, _, p, fpc = fuzz.cluster.cmeans(feats.T, num_clusters, 2, error=0.005, maxiter=1000)
    print(f"Model was trained finished training at {p} iterations. FPC={fpc} for {num_clusters} clusters.")
    return kmeans

def predict_kmeans(reader, in_filetype, rspecifier, kmeans, max_len = 3200):
    feats = []
    for utt, mat in file_reader_helper(rspecifier, in_filetype):
        if is_scipy_wav_style(mat):
            # If data is sound file, then got as Tuple[int, ndarray]
            rate, mat = mat
            mat = mat.astype(np.float64, order="C") / 32768.0
        nsample = len(mat)
        feat = reader.get_feats(mat, nsample).numpy()
        pad_width = int(max_len - feat.shape[0])

        feat = np.pad(feat, ((0, pad_width), (0,0)), 'constant', constant_values=((0,), (0,)))
        feats.append(feat)
    print("Extracted all feats", flush=True)
    feats = np.stack(feats, axis=0) # n x seq x dim
    feats = feats.reshape((feats.shape[0], -1)) # n x (seq x dim)

    #labels = kmeans.predict(feats)
    labels, _, _, _, _, _ = fuzz.cluster.cmeans_predict(feats.T, kmeans, 2, error=0.0001, maxiter=1000)
    return labels.T

def write_clusters(scp, out_f, labels):
    
    with open(out_f, 'w') as cluster_out:
        utt_ids = [line.split()[0] for line in open(scp).readlines()]
        assert len(utt_ids) == len(labels)
        for uttid, label in zip(utt_ids, labels):
            label_str = np.array2string(label)[1:-1].replace('\n', '')
            cluster_out.write(f"{uttid} {label_str}\n")

def main(args):
    np.random.seed(args.seed)
    reader = ESPnetModelFeatureReader('exp/asr_train_asr_conformer_hubert_raw_en_word/50epoch.pth')
    kmeans = learn_kmeans(
        reader,
        in_filetype=args.in_filetype,
        rspecifier=args.rspecifier,
        max_len=args.max_len,
        num_clusters=args.num_clusters
    )

    #train_labels = kmeans.labels_
    #write_clusters('dump/raw/train/wav.scp', 'dump/raw/train/clusters', train_labels)

    
    train_labels = predict_kmeans(reader,
        in_filetype=args.in_filetype,
        rspecifier='scp:dump/raw/train_en_de/wav.scp',
        kmeans=kmeans,
        max_len=args.max_len
    )

    write_clusters('dump/raw/train_en_de/wav.scp', f'dump/raw/train_en_de/clusters_{args.num_clusters}', train_labels)
    
    valid_labels = predict_kmeans(reader,
        in_filetype=args.in_filetype,
        rspecifier='scp:dump/raw/valid_german/wav.scp',
        kmeans=kmeans,
        max_len=args.max_len
    )

    write_clusters('dump/raw/valid_german/wav.scp', f'dump/raw/valid_german/clusters_{args.num_clusters}', valid_labels)

    valid_labels = predict_kmeans(reader,
        in_filetype=args.in_filetype,
        rspecifier='scp:dump/raw/test_german/wav.scp',
        kmeans=kmeans,
        max_len=args.max_len
    )

    write_clusters('dump/raw/test_german/wav.scp', f'dump/raw/test_german/clusters_{args.num_clusters}', valid_labels)

# usage: python3 local/kmeans.py scp:<wav.scp containing your audio>
# ex: python3 local/kmeans.py scp:/ocean/projects/cis210027p/wc6255/espnet_da/egs2/iemocap/ser1/dump/raw/valid/wav.scp
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    print(str(args))

    main(args)
