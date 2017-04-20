import os
import pandas as pd

def read_dataset(path):
    """Initialize a dataset from a path containing files called "train_stances.csv" and "train_bodies.csv".

    Parameters
    ----------
    path : str, bytes, or object implementing the os.PathLike protocol (default is None)
        The path to the directory containing the data files.

    Returns
    -------
    data : Pandas DataFrame
        The dataset at the provided path.
    """
    stances = pd.read_csv(os.path.join(path, 'train_stances.csv'))
    bodies = pd.read_csv(os.path.join(path, 'train_bodies.csv'))
    return stances.merge(bodies, on='Body ID')