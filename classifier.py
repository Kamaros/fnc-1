import os
from lightgbm import LGBMClassifier
from sklearn.externals import joblib

def train_classifier(features, labels):
    """Trains a gradient boosting classifier on the input features and labels.

    Classifiers are read from file if previously trained, or trained then cached otherwise.

    Parameters
    ----------
    features : Pandas DataFrame
        DataFrame containing the training features.
    labels : Pandas DataFrame
        DataFrame containing class labels.

    Returns
    -------
    classifier : LightGBM LGBMClassifier
        The trained classifier.
    """
    filename = 'caches/models/classifier.pkl'

    if not os.path.isfile(filename):
        classifier = LGBMClassifier(num_leaves=80, n_estimators=80, reg_alpha=0.05, reg_lambda=0.05, objective='multiclass')
        classifier.fit(features, labels)
        joblib.dump(classifier, filename)
    else:
        classifier = joblib.load(filename)

    return classifier