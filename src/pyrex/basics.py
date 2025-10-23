import pickle
import glob
import os
from scipy import interpolate
import statistics


def read_pkl(file_dir):

    with open(file_dir, "rb") as f:
        data = pickle.load(f)
    return data


def write_pkl(outfname, data_dict):

    f = open(outfname, "wb")
    pickle.dump(data_dict, f)
    f.close()


def checkIfDuplicates(listofElems):

    for elem in listofElems:
        if listofElems.count(elem) > 1:
            return True
    return False


def get_filename(
    dirfile="/home/amin/Projects/School/Masters/25_26-Thesis/pyrex/data/",
):

    r = 0
    os.chdir(dirfile)

    for file in glob.glob("*.pkl"):
        r += 1
    if r < 1:
        raise ValueError(
            "No *pkl files found in "
            + str(dirfile)
            + " . Please run 'example/traindata.py' to produce the train data."
        )
    if r > 1:
        raise ValueError(
            "Found "
            + str(r)
            + "*pkl files in "
            + dirfile
            + " . Please remove other *pkl files than the training data."
        )
    else:
        dfs = dirfile + str(file)

    return dfs


def interp1D(trainkey, trainval, testkey):

    newkey, newval = check_duplicate_training(trainkey, trainval)

    if testkey < min(trainkey) or testkey > max(trainkey):
        interp = interpolate.interp1d(newkey, newval, fill_value="extrapolate")
    else:
        interp = interpolate.interp1d(newkey, newval)
    return interp(testkey)


def check_duplicate_training(trainkey, trainval):

    d = {}
    newkey = []
    newval = []

    for a, b in zip(list(trainkey), list(trainval)):
        d.setdefault(a, []).append(b)

    for key in d:
        newkey.append(key)
        newval.append(statistics.median(d[key]))
    return newkey, newval


__all__ = [
    "read_pkl",
    "write_pkl",
    "checkIfDuplicates",
    "checkIfFilesExist",
    "interp1D",
    "check_duplicate_training",
]
