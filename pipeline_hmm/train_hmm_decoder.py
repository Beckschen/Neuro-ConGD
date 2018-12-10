#!/usr/bin/env python

import argparse

import h5py as h5
import numpy as np
import scipy.io as sio


def stationary_distribution(A):
    # Just compute it iteratively
    pi = np.ones(A.shape[0])
    pi /= pi.sum()

    pi_prime = A.dot(pi)
    pi_prime /= pi_prime.sum()
    while np.linalg.norm(pi - pi_prime) > 10**-9:
        pi = pi_prime
        pi_prime = A.dot(pi)
        pi_prime /= pi_prime.sum()

    return pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("out", help="Where to store the decoder")
    args = parser.parse_args()

    dataset_path = args.dataset
    out_path = args.out

    with h5.File(dataset_path, "r") as f:
        label_index = list(f["label_index"])

        labels = []

        def collect(name, obj):
            if isinstance(obj, h5.Group) and "labels" in obj:
                labels.append(np.array(obj["labels"]))

        f.visititems(collect)

    ndata = len(labels)

    labels = np.concatenate(labels, axis=0)

    nclasses = len(label_index) + 1
    A = np.zeros((nclasses, nclasses), np.float32)

    blank_filter = labels == -1
    nblanks = np.count_nonzero(blank_filter)

    for i in range(label_index):
        fltr = labels == i
        ninstances = int(np.count_nonzero(np.logical_xor(fltr, np.roll(fltr, 1))) / 2)

        A[i, -1] = ninstances / np.count_nonzero(fltr)
        A[i, i] = 1 - A[i, -1]
        A[-1, i] = ninstances / nblanks

    A[-1, -1] = 1 - np.sum(A[-1, :-1])

    pi = stationary_distribution(A)

    sio.savemat(out_path, {"pi": pi, "A": A})


if __name__ == "__main__":
    main()
