import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

def generate_hold_out_split(data, training=0.8):
    """Returns a split of the data where no article bodies in the training set appear in the hold-out set.
    
    Parameters
    ----------
    data : Pandas DataFrame
        The DataFrame to split.
    training : float, int, or None (default is 0.8)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the training set. If int, represents the absolute number of training samples.

    Returns
    -------
    train : Pandas DataFrame
        The generated training set.
    hold_out : Pandas DataFrame
        The generated hold-out set.
    """
    train_filename = 'caches/preprocessing/raw_training_data.h5'
    test_filename = 'caches/preprocessing/raw_testing_data.h5'

    if not os.path.isfile(train_filename) and not os.path.isfile(test_filename):
        body_ids = list(set(data['Body ID']))
        train_body_ids, test_body_ids = train_test_split(body_ids, train_size=training)
        training_data, testing_data = data[data['Body ID'].isin(train_body_ids)], data[data['Body ID'].isin(test_body_ids)]
        training_data.to_hdf(train_filename, 'training_data', complevel=9, complib='blosc')
        testing_data.to_hdf(test_filename, 'testing_data', complevel=9, complib='blosc')
    else:
        training_data, testing_data = pd.read_hdf(train_filename, 'training_data'), pd.read_hdf(test_filename, 'testing_data')

    return training_data, testing_data