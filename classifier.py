import os
from lightgbm import LGBMClassifier
from sklearn.externals import joblib

from utils.scoring import lgbm_score

def train_classifier(features, labels):
    filename = 'caches/models/classifier.pkl'

    if not os.path.isfile(filename):
        classifier = LGBMClassifier(boosting_type='gbdt', num_leaves=40, n_estimators=40, reg_alpha=0.05, reg_lambda=0.05, objective='multiclass', silent=False)
        classifier.fit(features, labels, eval_metric=lgbm_score)
        joblib.dump(classifier, filename)
    else:
        classifier = joblib.load(filename)

    return classifier