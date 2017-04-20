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