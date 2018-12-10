#!/usr/bin/env python

import argparse
import concurrent.futures
import os

import h5py as h5
import numpy as np
import pandas as pd


def ewm(data, dt, halflife):
    """Compute an exponentially-weighted mean over the columns of data.

    """
    # Perform the calculations in double because it involves numbers pretty
    # close to 1 and 0 (consider halflife=10**6)
    columns = data.columns
    data = data.values.astype(np.float64)
    alpha = np.exp(np.log(0.5) / halflife)

    for i in range(1, data.shape[0]):
        decay = np.power(alpha, dt[i])

        # Maybe the second formulation is more numerically stable since there
        # is no multiplication with a number very close to 0.
        # data[i] = (1 - decay) * data[i] + decay * data[i - 1]
        data[i] = data[i] + decay * (data[i - 1] - data[i])

    return pd.DataFrame(data=data.astype(np.float32), columns=columns)


def read_data(directory):
    events_path = os.path.join(directory, "events.csv")

    # Read everything as float32 except the timestamps
    columns = pd.read_csv(events_path, nrows=0).columns
    dtype = {c: np.float32 for c in columns}
    dtype["timestamp"] = np.int32
    events = pd.read_csv(events_path, dtype=dtype)

    # Preprocess events

    # Compute time since previous event
    ts = events["timestamp"].values
    dt = np.zeros(len(events), dtype=np.float32)
    dt[1:] = ts[1:] - ts[:-1]
    events["dt"] = dt

    # Compare events to an exponentially decaying mean position over the last
    # second to capture the major movement
    slow_mean = ewm(events[["x", "y"]], dt, halflife=10**6)
    events["dx-slow"] = events["x"] - slow_mean["x"]
    events["dy-slow"] = events["y"] - slow_mean["y"]

    # And also compare them to another mean over the last 50ms for quick
    # movements
    fast_mean = ewm(events[["x", "y"]], dt, halflife=50 * 10**3)
    events["dx-fast"] = events["x"] - fast_mean["x"]
    events["dy-fast"] = events["y"] - fast_mean["y"]

    # Drop columns with absolute values. We keep the timestamp because that is
    # needed for labeling.
    events = events.drop(["x", "y"], axis=1)

    # Reorder columns just to have them in the original order
    events = events[["timestamp", "dt", "dx-slow", "dx-fast", "dy-slow", "dy-fast", "polarity"]]

    # Split timestamps from data for further processing
    timestamps = events["timestamp"].values
    data = events.drop("timestamp", axis=1).values

    return timestamps, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--based-on", help="Dataset to reuse the normalization parameters from")
    parser.add_argument("dirs", nargs="*", help="File to save data to")
    parser.add_argument("out", help="HDF5 to write to")
    args = parser.parse_args()

    based_on = args.based_on
    directories = args.dirs
    out_path = args.out
    # 并发池！ 同时读取
    # 有没有可能把event_csv文件集中在一个路径中呢，不存在,因为join( dir + event.csv)
    with concurrent.futures.ProcessPoolExecutor() as ex:
        event_data = list(ex.map(read_data, directories))

    # 对应的read data return 的两个家伙
    timestamps = [t for t, d in event_data]
    data = [d for t, d in event_data]

    if based_on:
        # Reuse normalization parameters from another dataset (e.g. for a
        # validation set)
        with h5.File(based_on) as f:
            mu = np.array(f["/events"].attrs["mu"])
            sigma = np.array(f["/events"].attrs["sigma"])
    else:
        # Compute mean and standard deviation for normalization
        ndata = np.sum([d.shape[0] for d in data])
        with concurrent.futures.ThreadPoolExecutor() as ex:
            mu = np.sum(list(ex.map(lambda d: np.sum(d, axis=0, dtype=np.float64) / ndata, data)), axis=0)
            sigma = np.sqrt(np.sum(list(ex.map(lambda d: np.sum(np.square(d - mu), axis=0) / ndata, data)), axis=0))

        # We computed mu and sigma in double precision, but now convert them back
        # so that the data will not be converted
        mu = mu.astype(np.float32)
        sigma = sigma.astype(np.float32)

    # Normalize the data
    for i in range(len(data)):
        data[i] = (data[i] - mu) / sigma

    # Save the preprocessed data in an HDF5 file
    # Attributes support the metadata to describe data
    with h5.File(out_path, 'w') as f:
        # mu sigma这些属性是归为events的！
        # each is a numpy array containing 6 values
        grp = f.create_group("events")
        grp.attrs["mu"] = mu
        grp.attrs["sigma"] = sigma

        for directory, ts, d in zip(directories, timestamps, data):
            # return the last name of a directory, here "gerrit-1"
            name = os.path.basename(directory)
            dgrp = grp.create_group(name)
            # the attribute "directory" to "gerrit-1" is e:/deadline1115/paper/data/gerrit-1
            dgrp.attrs["directory"] = directory
            dgrp["timestamps"] = ts
            dgrp["data"] = d


if __name__ == "__main__":
    main()
