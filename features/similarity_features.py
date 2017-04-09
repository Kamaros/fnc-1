import numpy as np
import pandas as pd
from textacy.math_utils import cosine_similarity
from textacy.similarity import jaccard

from models import word2vec
from utils.notebook import in_notebook
from .feature import Feature

if in_notebook():
    from tqdm import tqdm_notebook
    tqdm = tqdm_notebook()
else:
    from tqdm import tqdm

class CosineSimilarity(Feature):
    """Cosine similarity between headline and article vector representations."""
    @staticmethod
    def cos_similarity(row):
        headline = row[0]
        body = row[1]
        return cosine_similarity(headline, body)

    @classmethod
    def cos_similarities(cls, dataframe):
        columns = dataframe.columns

        if len(columns) != 2:
            raise ValueError('CosineSimilarity expects input DataFrame to have exactly 2 columns containing vectors.')

        vectorization_method = columns[0].replace('Headline ', '')
        key = '{} Cosine Similarity'.format(vectorization_method)

        tqdm.pandas(desc='DataFrame -> CosSim')
        return pd.DataFrame({key: dataframe.progress_apply(cls.cos_similarity, axis=1, raw=True)})

    @classmethod 
    def get_feature_generator(cls):
        return cls.cos_similarities

class WMDSimilarity(Feature):
    """Word Mover's Distance between headline and article word vectors."""
    @staticmethod
    def wmd_similarities(dataframe):
        word2vec_model = word2vec()

        def wmd_similarity(row):
            headline = [word for word in row['Headline'].split() if word in word2vec_model.vocab]
            body = [word for word in row['articleBody'].split() if word in word2vec_model.vocab]
            wmd = word2vec_model.wmdistance(headline, body)
            if np.isinf(wmd):
                wmd = 2
            return wmd

        tqdm.pandas(desc='DataFrame -> WMD')
        return pd.DataFrame({'Word Mover\'s Distance': dataframe.progress_apply(wmd_similarity, axis=1)})

    @classmethod
    def get_feature_generator(cls):
        return cls.wmd_similarities

class WordOverlapSimilarity(Feature):
    """Jaccard distance, the overlap ratio of words contained in both the headline and footer."""
    @staticmethod
    def overlap(row):
        headline = row['Headline'].split()
        body = row['articleBody'].split()
        if len(headline) == 0 and len(body) == 0:
            return 0
        return jaccard(headline, body)

    @classmethod
    def word_overlap(cls, dataframe):
        tqdm.pandas(desc='DataFrame -> Overlap')
        return pd.DataFrame({'Word Overlap': dataframe.progress_apply(cls.overlap, axis=1)})

    @classmethod
    def get_feature_generator(cls):
        return cls.word_overlap