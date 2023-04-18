import argparse

import numpy as np

from feature_reader import (
    MfccFeatureReader
)

from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import is_scipy_wav_style
from sklearn.cluster import KMeans

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_len", type=int, default=6400)
    parser.add_argument("--seed", default=0, type=int)
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
    reader, in_filetype, rspecifier, write_num_frames=None, max_len = 6400
):
    feats = []
    i = 0
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

    feats = np.stack(feats, axis=0) # n x seq x dim
    feats = feats.reshape((feats.shape[0], -1)) # n x (seq x dim)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=1).fit(feats)
    labels = kmeans.labels_ 
    print(labels)

def main(args):
    np.random.seed(args.seed)
    reader = MfccFeatureReader(sample_rate=args.sample_rate)

    learn_kmeans(
        reader,
        in_filetype=args.in_filetype,
        rspecifier=args.rspecifier,
        max_len=args.max_len
    )

# usage: python3 local/kmeans.py scp:<wav.scp containing your audio>
# ex: python3 local/kmeans.py scp:/ocean/projects/cis210027p/wc6255/espnet_da/egs2/iemocap/ser1/dump/raw/valid/wav.scp
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    print(str(args))

    main(args)
