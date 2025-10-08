import pickle
import glob
import os
from scipy import interpolate
import statistics


def read_pkl(file_dir):
    """
    Read a pickle file.
    Parameters
    ----------
    file_dir    : The directory of the file.

    Returns
    ------
    f           : The read file with keys.
    """
    with open(file_dir, "rb") as f:
        data = pickle.load(f)
    return data


def write_pkl(outfname, data_dict):
    """
    Write a pickle file.
    Parameters
    ----------
    outfname  : The directory of the output file.
    data_dict : The data variables to be written.

    Returns
    ------
    The written file with keys in outfname.
    """

    f = open(outfname, "wb")
    pickle.dump(data_dict, f)
    f.close()


def checkIfDuplicates(listofElems):
    """
    Check if the given list contains any duplicates.
    """
    for elem in listofElems:
        if listofElems.count(elem) > 1:
            return True
    return False


def checkIfFilesExist(
    dirfile="/home/amin/Projects/School/Masters/25_26-Thesis/pyrex/data/",
):
    """
    Check if pickle files exist.
    """
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
    """
    Perform 1D interpolation.

    Parameters
    ----------
    trainkey: []
            Array of the x interpolation values.
    trainval: []
            Array of the y interpolation values.
    testkey: []
            The position of the new x for interpolation.

    Returns
    ------
    result: []
            The interpolated value in 1 dimension.

    """
    newkey, newval = check_duplicate_training(trainkey, trainval)

    if testkey < min(trainkey) or testkey > max(trainkey):
        interp = interpolate.interp1d(newkey, newval, fill_value="extrapolate")
        result = interp(testkey)
    else:
        interp = interpolate.interp1d(trainkey, trainval)
        result = interp(testkey)
    return result


def check_duplicate_training(trainkey, trainval):
    """
    Check if the training keys have duplicate numbers.
    If so, get its average values before performing interpolation.
    Parameters
    ----------
    trainkey: []
            Array of the x interpolation values.
    trainval: {float}
            Array of the y interpolation values.

    Returns
    ------
    newkey: []
            The new x interpolation values (no duplicates).
    newval: []
            The new y interpolation values, average of the old trainval with duplicate trainkey.

    """
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
