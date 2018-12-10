#!/usr/bin/env python

import argparse
import concurrent.futures
import os

import h5py as h5
import numpy as np
import scipy.io as sio


def viterbi(logits, pi, log_A, softmax=True):
    if softmax:
        # Softmax to convert logits to probabilities
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= np.sum(p, axis=1)[:, np.newaxis]
    else:
        p = logits

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
        label_index = list(f["label_index"])

        directories = []
        timestamps = []
        data = []

        def collect(name, obj):
            if isinstance(obj, h5.Group) and "directory" in obj.attrs and "timestamps" in obj and "data" in obj:
                directories.append(obj.attrs["directory"])
                timestamps.append(np.array(obj["timestamps"]))
                data.append(np.array(obj["data"]))

        f.visititems(collect)

    log_A = np.log(A)
    with concurrent.futures.ProcessPoolExecutor() as ex:
        zs = list(ex.map(viterbi, data, [pi] * len(data), [log_A] * len(data), [softmax] * len(data)))

    label_index.append("<blank>")

    with h5.File(out_path, "w") as f:
        coll = f.create_group("hmm")

        for d, t, z in zip(directories, timestamps, zs):
            onehot = np.zeros((len(z), data[0].shape[1]), np.float32)
            onehot[np.arange(len(z)), z] = 1.0

            name = os.path.basename(d)
            grp = coll.create_group(name)
            grp.attrs["directory"] = d
            grp["timestamps"] = t
            grp["data"] = onehot

            print(name)
            prev = None
            for Z in z:
                if prev is None or prev != Z:
                    print(label_index[Z])
                    prev = Z



if __name__ == "__main__":
    main()
