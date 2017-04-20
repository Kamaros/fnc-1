import os
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_hold_out_split(data, test_size=0.1):
    """Returns a split of the data where no article bodies in the training set appear in the hold-out set.
    
    Parameters
    ----------
    data : Pandas DataFrame
        The DataFrame to split.
    test_size : float, int, or None (default is 0.8)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test set. If int, represents the absolute number of test samples.

    Returns
    -------
    train : Pandas DataFrame
        The generated training set.
    hold_out : Pandas DataFrame
        The generated hold-out set.
    unused : Pandas DataFrame
        Unused training examples sharing headlines with test samples.
    """
    # train_filename = 'caches/preprocessing/raw_training_data.h5'
    # test_filename = 'caches/preprocessing/raw_testing_data.h5'
    # unused_filename = 'caches/preprocessing/raw_unused_data.h5'

    # if not os.path.isfile(train_filename) and not os.path.isfile(test_filename) and not os.path.isfile(unused_filename):
        # Split unique body ids into training and testing sets
    body_ids = list(set(data['Body ID']))
    train_body_ids, test_body_ids = train_test_split(body_ids, test_size=test_size)
    training_data, testing_data = data[data['Body ID'].isin(train_body_ids)], data[data['Body ID'].isin(test_body_ids)]

    # Remove headlines present in the testing set from the training set to prevent headline bleeding
    testing_headlines = list(set(testing_data['Headline']))
    training_data, unused_data = training_data[~training_data['Headline'].isin(testing_headlines)], training_data[training_data['Headline'].isin(testing_headlines)]

        # training_data.to_hdf(train_filename, 'training_data', complevel=9, complib='blosc')
        # testing_data.to_hdf(test_filename, 'testing_data', complevel=9, complib='blosc')
        # unused_data.to_hdf(unused_filename, 'unused_data', complevel=9, complib='blosc')
    # else:
    #     training_data, testing_data, unused_data = pd.read_hdf(train_filename, 'training_data'), pd.read_hdf(test_filename, 'testing_data'), pd.read_hdf(unused_filename, 'unused_data')

    return training_data, testing_data, unused_data