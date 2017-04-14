import os
from lightgbm import LGBMClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

from utils.scoring import lgbm_score

def train_classifier(features, labels):
    filename = 'caches/models/classifier.pkl'

    if not os.path.isfile(filename):
        param_grid = [{
            'boosting_type': ['gbdt', 'dart'],
            'num_leaves': [20, 30, 40],
            'n_estimators': [20, 30, 40],
            'reg_alpha': [0.05, 0.1, 0.15],
            'reg_lambda': [0.05, 0.1, 0.15]
        }]
        classifier = GridSearchCV(LGBMClassifier(objective='multiclass', silent=False), 
            param_grid=param_grid, 
            fit_params={'eval_metric': lgbm_score})
        classifier.fit(features, labels)
        joblib.dump(classifier, filename)
    else:
        classifier = joblib.load(filename)

    return classifier