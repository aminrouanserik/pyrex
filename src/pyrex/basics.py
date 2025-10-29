import pickle
import glob
import os
from scipy import interpolate
import statistics


def read_pkl(file_dir: str) -> any:
    """Returns data from a specified pickle file.

    Args:
        file_dir (str): Directory to read data from.

    Returns:
        any: Data in the file.
    """
    with open(file_dir, "rb") as f:
        data = pickle.load(f)
    return data


def write_pkl(outfname: str, data_dict: dict) -> None:
    """Writes a dictionary to a pickle file in a specified directory.

    Args:
        outfname (str): Path to the file to write to.
        data_dict (dict): Dictionary to write to the file.
    """
    f = open(outfname, "wb")
    pickle.dump(data_dict, f)
    f.close()


def get_filename(
    dirfile: str = "/home/amin/Projects/School/Masters/25_26-Thesis/pyrex/data/",
) -> str:
    """Get the filename of the pickle data file.

    Args:
        dirfile (str, optional): Directory in which pickle file is stored. Defaults to "/home/amin/Projects/School/Masters/25_26-Thesis/pyrex/data/".

    Raises:
        ValueError: Raised if there are no pickle files in the directory,
        ValueError: Raised if more than one pickle file was found.

    Returns:
        str: Path to the pickle file.
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


def interp1D(
    trainkey: list[float], trainval: list[float], testkey: list[float]
) -> list[float]:
    """Interpolates a key between various points in the training data.

    Args:
        trainkey (list[float]): List of values in a grid on which the interpolation is based.
        trainval (list[float]): Values corresponding to the train grid.
        testkey (list[float]): List of values in a grid of which to get the interpolant.

    Returns:
        list[float]: The interpolateed data.
    """
    newkey, newval = check_duplicate_training(trainkey, trainval)

    if testkey < min(trainkey) or testkey > max(trainkey):
        interp = interpolate.interp1d(newkey, newval, fill_value="extrapolate")
    else:
        interp = interpolate.interp1d(trainkey, trainval)
    return interp(testkey)


def check_duplicate_training(
    trainkey: list[float], trainval: list[float]
) -> tuple[list[float], list[float]]:
    """Checks whether there is any duplicate keys and calculates the median.

    Args:
        trainkey (list[float]): List of values in a grid on which the interpolation is based.
        trainval (list[float]): Values corresponding to the train grid.

    Returns:
        tuple[list[float], list[float]]: Tuple of the new list of values in a grid on which the interpolation is based and
        the values corresponding to that grid.
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
