import pandas as pd
from sklearn.model_selection import train_test_split

def generate_hold_out_split(data, test_size=0.1, random_state=20170206):
    """Returns a split of the data where no article bodies in the training set appear in the hold-out set.
    
    Parameters
    ----------
    data : Pandas DataFrame
        The DataFrame to split.
    test_size : float, int, or None (default is 0.8)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test set. If int, represents the absolute number of test samples.
    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    Returns
    -------
    train : Pandas DataFrame
        The generated training set.
    hold_out : Pandas DataFrame
        The generated hold-out set.
    unused : Pandas DataFrame
        Unused training examples sharing headlines with test samples.
    """
    # Split unique body ids into training and testing sets
    body_ids = list(set(data['Body ID']))
    train_body_ids, test_body_ids = train_test_split(body_ids, test_size=test_size, random_state=random_state)
    training_data, testing_data = data[data['Body ID'].isin(train_body_ids)], data[data['Body ID'].isin(test_body_ids)]

    # Remove headlines present in the testing set from the training set to prevent headline bleeding
    testing_headlines = list(set(testing_data['Headline']))
    training_data, unused_data = training_data[~training_data['Headline'].isin(testing_headlines)], training_data[training_data['Headline'].isin(testing_headlines)]

    return training_data, testing_data, unused_data