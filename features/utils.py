import os
import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.externals import joblib

from .sentiment_features import PolarityScorer, EmotionScorer
from .similarity_features import CosineSimilarity, WMDSimilarity, WordOverlapSimilarity
from .vectorizer_features import BoWVectorizer, TfidfVectorizer, LSIVectorizer, RPVectorizer, LDAVectorizer, WordVectorCentroidVectorizer, Doc2VecVectorizer

def extract_features(data, raw_data):
    """Extracts features from datasets.

    Features are read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing preprocessed text data.
    raw_data : Pandas DataFrame
        DataFrame containing unprocessed text data.

    Returns
    -------
    features : Pandas DataFrame
        DataFrame containing features extracted from the datasets.
    """
    filename = 'caches/features/features.pkl'

    if not os.path.isfile(filename):
        bow = BoWVectorizer.transform(data)
        bow_cos = CosineSimilarity.transform(bow, key='bow')

        tfidf = TfidfVectorizer.transform(data)
        tfidf_cos = CosineSimilarity.transform(tfidf, key='tfidf')

        lsi = LSIVectorizer.transform(data)
        lsi_cos = CosineSimilarity.transform(lsi, key='lsi')

        rp = RPVectorizer.transform(data)
        rp_cos = CosineSimilarity.transform(rp, key='rp')

        lda = LDAVectorizer.transform(data)
        lda_cos = CosineSimilarity.transform(lda, key='lda')

        w2v = WordVectorCentroidVectorizer.transform(data)
        w2v_cos = CosineSimilarity.transform(w2v, key='w2v')

        d2v = Doc2VecVectorizer.transform(data)
        d2v_cos = CosineSimilarity.transform(d2v, key='d2v')

        wmd = WMDSimilarity.transform(data)

        overlap = WordOverlapSimilarity.transform(data)

        polarities = PolarityScorer.transform(raw_data)

        emotion = EmotionScorer.transform(data)

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
    columns = []
    for column in features.columns:
        head = features[column][0]
        if isinstance(head, np.ndarray):
            columns += ['{} {}'.format(column, i) for i in range(head.size)]
        else:
            columns.append(column)
    column_matrices = [np.asmatrix(np.stack(features[column].values)) for column in features.columns]
    column_matrices = [m.T if m.shape[0] == 1 else m for m in column_matrices]
    column_matrices = np.concatenate(column_matrices, axis=1)
    return pd.DataFrame(column_matrices, index=features.index, columns=columns)