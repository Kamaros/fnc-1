import numpy as np
import pandas as pd
from gensim.matutils import sparse2full

from models import dictionary_corpus, hashdictionary_corpus, tfidf, lsi, rp, lda, word2vec, doc2vec
from utils.notebook import in_notebook
from .feature import Feature

if in_notebook():
    from tqdm import tqdm_notebook
    tqdm = tqdm_notebook()
else:
    from tqdm import tqdm

class BoWVectorizer(Feature):
    """Bag of Words vector transformer."""
    @staticmethod
    def bow_vectors(dataframe):
        dimensionality = 2000
        dictionary = hashdictionary_corpus(dataframe, id_range=dimensionality)

        def bow_vector(doc):
            doc_bow = dictionary.doc2bow(doc.split())
            return sparse2full(doc_bow, length=dimensionality)

        tqdm.pandas(desc='Headline -> BoW')
        headline_bow = dataframe['Headline'].progress_apply(bow_vector)

        tqdm.pandas(desc='articleBody -> BoW')
        body_bow = dataframe['articleBody'].progress_apply(bow_vector)

        return pd.DataFrame({'Headline BoW': headline_bow, 'articleBody BoW': body_bow})

    @classmethod
    def get_feature_generator(cls):
        return cls.bow_vectors

class TfidfVectorizer(Feature):
    """tf-idf vector transformer."""
    @staticmethod
    def tfidf_vectors(dataframe):
        dimensionality = 2000
        dictionary = hashdictionary_corpus(dataframe, id_range=dimensionality)
        tfidf_model = tfidf(dataframe, max_words=dimensionality)

        def tfidf_vector(doc):
            doc_bow = dictionary.doc2bow(doc.split())
            doc_tfidf = tfidf_model[doc_bow]
            return sparse2full(doc_tfidf, length=dimensionality)

        tqdm.pandas(desc='Headline -> tf-idf')
        headline_tfidf = dataframe['Headline'].progress_apply(tfidf_vector)

        tqdm.pandas(desc='articleBody -> tf-idf')
        body_tfidf = dataframe['articleBody'].progress_apply(tfidf_vector)

        return pd.DataFrame({'Headline tf-idf': headline_tfidf, 'articleBody tf-idf': body_tfidf})

    @classmethod
    def get_feature_generator(cls):
        return cls.tfidf_vectors

class LSIVectorizer(Feature):
    """Latent Semantic Indexing vector transformer."""
    @staticmethod
    def lsi_vectors(dataframe):
        dictionary = dictionary_corpus(dataframe)
        tfidf_model = tfidf(dataframe)
        lsi_model = lsi(dataframe)

        def lsi_vector(doc):
            doc_bow = dictionary.doc2bow(doc.split())
            doc_tfidf = tfidf_model[doc_bow]
            doc_lsi = lsi_model[doc_tfidf]
            return sparse2full(doc_lsi, length=lsi_model.num_topics)

        tqdm.pandas(desc='Headline -> LSI')
        headline_lsi = dataframe['Headline'].progress_apply(lsi_vector)

        tqdm.pandas(desc='articleBody -> LSI')
        body_lsi = dataframe['articleBody'].progress_apply(lsi_vector)

        return pd.DataFrame({'Headline LSI': headline_lsi, 'articleBody LSI': body_lsi})

    @classmethod
    def get_feature_generator(cls):
        return cls.lsi_vectors

class RPVectorizer(Feature):
    """Random Projection vector transformer."""
    @staticmethod
    def rp_vectors(dataframe):
        dictionary = dictionary_corpus(dataframe)
        tfidf_model = tfidf(dataframe)
        rp_model = rp(dataframe)

        def rp_vector(doc):
            doc_bow = dictionary.doc2bow(doc.split())
            doc_tfidf = tfidf_model[doc_bow]
            doc_rp = rp_model[doc_tfidf]
            return sparse2full(doc_rp, length=rp_model.num_topics)

        tqdm.pandas(desc='Headline -> RP')
        headline_rp = dataframe['Headline'].progress_apply(rp_vector)

        tqdm.pandas(desc='articleBody -> RP')
        body_rp = dataframe['articleBody'].progress_apply(rp_vector)

        return pd.DataFrame({'Headline RP': headline_rp, 'articleBody RP': body_rp})

    @classmethod
    def get_feature_generator(cls):
        return cls.rp_vectors

class LDAVectorizer(Feature):
    """Latent Dirichlet Allocation vector transformer."""
    @staticmethod
    def lda_vectors(dataframe):
        dictionary = dictionary_corpus(dataframe)
        lda_model = lda(dataframe)

        def lda_vector(doc):
            doc_bow = dictionary.doc2bow(doc.split())
            doc_lda = lda_model[doc_bow]
            return sparse2full(doc_lda, length=lda_model.num_topics)

        tqdm.pandas(desc='Headline -> LDA')
        headline_lda = dataframe['Headline'].progress_apply(lda_vector)

        tqdm.pandas(desc='articleBody -> LDA')
        body_lda = dataframe['articleBody'].progress_apply(lda_vector)

        return pd.DataFrame({'Headline LDA': headline_lda, 'articleBody LDA': body_lda})

    @classmethod
    def get_feature_generator(cls):
        return cls.lda_vectors

class WordVectorCentroidVectorizer(Feature):
    """word2vec centroid transformer."""
    @staticmethod
    def word_vector_centroids(dataframe):
        word2vec_model = word2vec()

        def word_vector_centroid(doc):
            valid_words = [word for word in doc if word in word2vec_model.vocab]
            if len(valid_words) == 0:
                return np.zeros(300)
            return np.mean(word2vec_model[valid_words], axis=0)

        tqdm.pandas(desc='Headline -> word2vec')
        headline_word2vec = dataframe['Headline'].progress_apply(word_vector_centroid)

        tqdm.pandas(desc='articleBody -> word2vec')
        body_word2vec = dataframe['articleBody'].progress_apply(word_vector_centroid)

        return pd.DataFrame({'Headline Mean Word Vector': headline_word2vec, 'articleBody Mean Word Vector': body_word2vec})

    @classmethod
    def get_feature_generator(cls):
        return cls.word_vector_centroids

class Doc2VecVectorizer(Feature):
    """doc2vec transformer."""
    @staticmethod
    def doc_vectors(dataframe):
        doc2vec_model = doc2vec(dataframe)

        def doc_vector(doc):
            valid_words = [word for word in doc.split() if word in doc2vec_model.wv.vocab]
            return doc2vec_model.infer_vector(valid_words, steps=20)

        tqdm.pandas(desc='Headline -> doc2vec')
        headline_doc2vec = dataframe['Headline'].progress_apply(doc_vector)

        tqdm.pandas(desc='articleBody -> doc2vec')
        body_doc2vec = dataframe['articleBody'].progress_apply(doc_vector)

        return pd.DataFrame({'Headline doc2vec': headline_doc2vec, 'articleBody doc2vec': body_doc2vec})

    @classmethod
    def get_feature_generator(cls):
        return cls.doc_vectors