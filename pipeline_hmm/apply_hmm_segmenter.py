#!/usr/bin/env python

import argparse
import concurrent.futures
import os

import h5py as h5
import numpy as np
import scipy.io as sio


def viterbi(p, pi, log_A):
    log_delta = np.log(p)
    log_delta[0] += np.log(pi)

    a = np.zeros(log_delta.shape, np.int8)

    for t in range(1, len(p)):
        for j in range(log_delta.shape[1]):
            a[t, j] = np.argmax(log_delta[t - 1] + log_A[:, j])
            log_delta[t, j] += log_delta[t - 1, a[t, j]] + log_A[a[t, j], j]

    z = np.empty(len(log_delta), np.int8)
    z[-1] = np.argmax(log_delta[-1])

    for t in range(len(z) - 2, -1, -1):
        z[t] = a[t + 1, z[t + 1]]

    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", default=False, action="store_true", dest="probabilities", help="The features are already probabilities (do not apply softmax)")
    parser.add_argument("hmm", help="Path to trained HMM")
    parser.add_argument("dataset", help="Path to classifications")
    parser.add_argument("out", help="Where to store decodings")
    args = parser.parse_args()

    softmax = not args.probabilities
    hmm_path = args.hmm
    dataset_path = args.dataset
    out_path = args.out

    values = sio.loadmat(hmm_path)
    pi = np.squeeze(values["pi"])
    A = values["A"]

    with h5.File(dataset_path, "r") as f:
        directories = []
        timestamps = []
        data = []

        def collect(name, obj):
            if isinstance(obj, h5.Group) and "directory" in obj.attrs and "timestamps" in obj and "data" in obj:
                directories.append(obj.attrs["directory"])
                timestamps.append(np.array(obj["timestamps"]))
                data.append(np.array(obj["data"]))

        f.visititems(collect)

    # Add up probabilities of all non-blank classes
    binary_data = []
    for d in data:
        # Softmax to convert logits to probabilities
        if softmax:
            d = np.exp(d - np.max(d, axis=1, keepdims=True))
            d /= np.sum(d, axis=1)[:, np.newaxis]

        binary_data.append(np.stack([np.sum(d[:, :-1], axis=1), d[:, -1]], axis=-1))

    log_A = np.log(A)
    with concurrent.futures.ProcessPoolExecutor() as ex:
        zs = list(ex.map(viterbi, binary_data, [pi] * len(data), [log_A] * len(data)))

    # Compute segment boundaries
    bounds = []
    for z in zs:
        fltr = z == 0
        indices = np.nonzero(np.logical_xor(fltr, np.roll(fltr, 1)))[0]
        bounds.append(list(map(tuple, zip(indices[0::2], indices[1::2]))))

    # Filter out segments < 500ms
    bounds = [[(s, e) for s, e in b if t[e] - t[s] >= 0.5 * 10**6]
              for t, b in zip(timestamps, bounds)]

    with h5.File(out_path, "w") as f:
        coll = f.create_group("hmm")

        for dir, t, b, d in zip(directories, timestamps, bounds, data):
            # Compute MAP for each segment
            cls = np.full(len(d), d.shape[1] - 1)

            for s, e in b:
                cls[s:e] = d[s:e, :-1].sum(axis=0).argmax()

            onehot = np.zeros((len(cls), d.shape[1]), np.float32)
            onehot[np.arange(len(cls)), cls] = 1.0

            name = os.path.basename(dir)
            grp = coll.create_group(name)
            grp.attrs["directory"] = dir
            grp["timestamps"] = t
            grp["data"] = onehot


if __name__ == "__main__":
    main()
