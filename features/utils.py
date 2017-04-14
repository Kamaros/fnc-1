import os
import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MaxAbsScaler

from .sentiment_features import PolarityScorer, EmotionScorer
from .similarity_features import CosineSimilarity, WMDSimilarity, WordOverlapSimilarity
from .vectorizer_features import BoWVectorizer, TfidfVectorizer, LSIVectorizer, RPVectorizer, LDAVectorizer, WordVectorCentroidVectorizer, Doc2VecVectorizer

def extract_features(data, raw_data, key):
    """Extracts features from datasets.

    Features are read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing preprocessed text data.
    raw_data : Pandas DataFrame
        DataFrame containing unprocessed text data.
    key : str
        Key used for caching.

    Returns
    -------
    features : Pandas DataFrame
        DataFrame containing features extracted from the datasets.
    """
    filename = 'caches/features/{}_features.pkl'.format(key)

    if not os.path.isfile(filename):
        bow = BoWVectorizer.transform(data, '{}_data'.format(key))
        bow_cos = CosineSimilarity.transform(bow, '{}_bow'.format(key))

        tfidf = TfidfVectorizer.transform(data, '{}_data'.format(key))
        tfidf_cos = CosineSimilarity.transform(tfidf, '{}_tfidf'.format(key))

        lsi = LSIVectorizer.transform(data, '{}_data'.format(key))
        lsi_cos = CosineSimilarity.transform(lsi, '{}_lsi'.format(key))

        rp = RPVectorizer.transform(data, '{}_data'.format(key))
        rp_cos = CosineSimilarity.transform(rp, '{}_rp'.format(key))

        lda = LDAVectorizer.transform(data, '{}_data'.format(key))
        lda_cos = CosineSimilarity.transform(lda, '{}_lda'.format(key))

        w2v = WordVectorCentroidVectorizer.transform(data, '{}_data'.format(key))
        w2v_cos = CosineSimilarity.transform(w2v, '{}_w2v'.format(key))

        d2v = Doc2VecVectorizer.transform(data, '{}_data'.format(key))
        d2v_cos = CosineSimilarity.transform(d2v, '{}_d2v'.format(key))

        wmd = WMDSimilarity.transform(data, '{}_data'.format(key))

        overlap = WordOverlapSimilarity.transform(data, '{}_data'.format(key))

        polarities = PolarityScorer.transform(raw_data, '{}_data'.format(key))

        emotion = EmotionScorer.transform(data, '{}_data'.format(key))

        features = pd.concat([
            bow,
            bow_cos,
            tfidf,
            tfidf_cos,
            lsi,
            lsi_cos,
            rp,
            rp_cos,
            lda,
            lda_cos,
            w2v,
            w2v_cos,
            d2v,
            d2v_cos,
            wmd,
            overlap,
            polarities,
            emotion
        ], axis=1)
        features.to_pickle(filename)
    else:
        features = pd.read_pickle(filename)

    return features

def flatten_features(features):
    """Flattens a DataFrame of features so that columns containing arrays are split into separate columns.

    Parameters
    ----------
    features : Pandas DataFrame
        DataFrame of features.

    Returns
    -------
    flattened : Pandas DataFrame
        Flattened DataFrame of features.
    """    
    column_matrices = [np.asmatrix(np.stack(features[column].values)) for column in features.columns]
    column_matrices = [m.T if m.shape[0] == 1 else m for m in column_matrices]
    column_matrices = np.concatenate(column_matrices, axis=1)
    return pd.DataFrame(column_matrices, index=features.index)

def scale_features(features):
    """Scales features in a DataFrame by the absolute value of the maximum value in each column, thus preserving sparsity.

    The scaling model is read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    features : Pandas DataFrame
        DataFrame of features.

    Returns
    -------
    scaled : Pandas DataFrame
        Scaled DataFrame of features.
    """
    filename = 'caches/preprocessing/scaler.pkl'

    if not os.path.isfile(filename):
        scaler = MaxAbsScaler()
        scaler.fit(features)
        joblib.dump(scaler, filename)
    else:
        scaler = joblib.load(filename)

    return pd.DataFrame(scaler.transform(features), index=features.index)

def decompose_features(features):
    """Applies PCA to features in a DataFrame, preserving 95% of variance information.

    The transformation is read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    features : Pandas DataFrame
        DataFrame of features.

    Returns
    -------
    decomposed : Pandas DataFrame
        Decomposed DataFrame of features.
    """
    filename = 'caches/preprocessing/pca.pkl'

    if not os.path.isfile(filename):
        pca = PCA(n_components=0.95, svd_solver='full')
        pca.fit(features)
        joblib.dump(pca, filename)
    else:
        pca = joblib.load(filename)

    return pd.DataFrame(pca.transform(features), index=features.index)

def select_features(features, labels=None):
    """Selects features DataFrame, keeping features with better than mean feature importance.

    Feature importance is calculated using an untuned LightGBM classifier.

    The transformation is read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    features : Pandas DataFrame
        DataFrame of features.

    Returns
    -------
    decomposed : Pandas DataFrame
        Decomposed DataFrame of features.
    """    
    filename = 'caches/preprocessing/feature_selector.pkl'

    if not os.path.isfile(filename):
        if labels == None:
            raise ValueError('select_features requires that the labels parameter be set if no preexisting model is found')
        feature_selector = SelectFromModel(LGBMClassifier(objective='multiclass', silent=False))
        feature_selector.fit(features, labels)
        joblib.dump(feature_selector, filename)
    else:
        feature_selector = joblib.load(filename)

    return pd.DataFrame(feature_selector.transform(features), index=features.index)