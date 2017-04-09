import os
import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, path=None, data=None):
        """Initialize a dataset from a path containing files called "train_stances.csv" and "train_bodies.csv", or from raw data.

        Parameters
        ----------
        path : str, bytes, or object implementing the os.PathLike protocol (default is None)
            The path to the directory containing the data files.
        data : Pandas DataFrame (default is None)
            Raw data to initialize the instance with.
        """
        if path is not None and data is None:
            stances = pd.read_csv(os.path.join(path, 'train_stances.csv'))
            bodies = pd.read_csv(os.path.join(path, 'train_bodies.csv'))
            self.data = stances.merge(bodies, on='Body ID')
        elif data is not None and path is None:
            self.data = data
        else:
            raise ValueError('Invalid parameters for Dataset constructor. Expected one of either path or data arguments to be set.')

    def generate_hold_out_split(self, training=0.8, random_state=25042017):
        """Returns a split of the data where no article bodies in the training set appear in the hold-out set.
        
        Parameters
        ----------
        training : float, int, or None (default is 0.8)
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the training set. If int, represents the absolute number of training samples.
        random_state : int or RandomState
            Pseudo-random number generator state used for random sampling.

        Returns
        -------
        train : Dataset
            The generated training set.
        hold_out : Dataset
            The generated hold-out set.
        """
        body_ids = list(set(self.data['Body ID']))
        train_body_ids, test_body_ids = train_test_split(body_ids, train_size=training, random_state=random_state)
        training_data, hold_out_data = self.data[self.data['Body ID'].isin(train_body_ids)], self.data[self.data['Body ID'].isin(test_body_ids)]
        return Dataset(data=training_data), Dataset(data=hold_out_data)