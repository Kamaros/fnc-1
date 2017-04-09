import os
import multiprocessing
import pickle
import wget
from gensim.corpora import Dictionary, HashDictionary
from gensim.models import LdaModel, LsiModel, KeyedVectors, RpModel, TfidfModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def text_corpus(dataframe):
    """Returns a list of unique documents stored in a DataFrame.

    Precomputed corpuses are read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        The DataFrame containing the documents to process.

    Returns
    -------
    corpus : list(str)
        List of unique documents stored in the DataFrame.
    """
    filename = 'caches/models/corpus.model'

    if not os.path.isfile(filename):
        corpus = set(dataframe['Headline'].values).union(dataframe['articleBody'].values)
        corpus = [doc.split() for doc in corpus]
        pickle.dump(corpus, open(filename, 'wb'))
    else:
        corpus = pickle.load(open(filename, 'rb'))

    return corpus

def dictionary_corpus(dataframe):
    """Returns a Dictionary mapping words to ids.

    Precomputed Dictionaries are read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        The DataFrame containing the documents to process.

    Returns
    -------
    dictionary : Gensim Dictionary
        Dictionary mapping words to ids.
    """
    filename = 'caches/models/dictionary.model'

    if not os.path.isfile(filename):
        corpus = text_corpus(dataframe)
        dictionary = Dictionary(corpus)
        dictionary.save(filename)
    else:
        dictionary = Dictionary.load(filename)

    return dictionary

def hashdictionary_corpus(dataframe, id_range=32000):
    """Returns a HashDictionary mapping words to ids.

    Precomputed HashDictionaries are read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        The DataFrame containing the documents to process.
    id_range : int
        The maximum number of ids available.

    Returns
    -------
    dictionary : Gensim HashDictionary
        HashDictionary mapping words to ids.
    """
    filename = 'caches/models/dictionary_{}.model'.format(id_range)

    if not os.path.isfile(filename):
        corpus = text_corpus(dataframe)
        dictionary = HashDictionary(corpus, id_range=id_range)
        dictionary.save(filename)
    else:
        dictionary = HashDictionary.load(filename)

    return dictionary

def bow_corpus(dataframe):
    """Returns a list of BoW vectors corresponding to documents stored in a DataFrame.

    Precomputed BoW vectors are read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        The DataFrame containing the documents to process.

    Returns
    -------
    corpus : list(list((int,int)))
        List of BoW vectors corresponding documents stored in the DataFrame.
    """
    filename = 'caches/models/bow.model'

    if not os.path.isfile(filename):
        corpus = text_corpus(dataframe)
        dictionary = dictionary_corpus(dataframe)
        bow = [dictionary.doc2bow(doc) for doc in corpus]
        pickle.dump(bow, open(filename, 'wb'))
    else:
        bow = pickle.load(open(filename, 'rb'))

    return bow    

def tfidf(dataframe, max_words=None):
    """Returns a tf-idf model for documents stored in a DataFrame.

    Precomputed models are read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        The DataFrame containing the documents to process.
    max_words : int (default is 2000000)
        The maximum number of words stored by the model.

    Returns
    -------
    model : Gensim TfidfModel
        tf-idf model for documents stored in the DataFrame.
    """
    suffix = '_{}'.format(max_words) if max_words else ''
    filename = 'caches/models/tfidf{}.model'.format(suffix)

    if not os.path.isfile(filename):
        if max_words:
            dictionary = hashdictionary_corpus(dataframe, id_range=max_words)
        else:
            dictionary = dictionary_corpus(dataframe)
        tfidf_model = TfidfModel(dictionary=dictionary)
        tfidf_model.save(filename)
    else:
        tfidf_model = TfidfModel.load(filename)

    return tfidf_model

def lsi(dataframe, num_topics=300):
    """Returns an LSI model for documents stored in a DataFrame.

    Precomputed models are read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        The DataFrame containing the documents to process.
    num_topics : int (default is 300)
        The number of topics to train the LSI model with.

    Returns
    -------
    model : Gensim LsiModel
        LSI model for documents stored in the DataFrame.
    """
    filename = 'caches/models/lsi.model'

    if not os.path.isfile(filename):
        dictionary = dictionary_corpus(dataframe)
        bow = bow_corpus(dataframe)
        tfidf_model = tfidf(dataframe)
        tfidf_corpus = tfidf_model[bow]
        lsi_model = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=num_topics)
        lsi_model.save(filename)
    else:
        lsi_model = LsiModel.load(filename)

    return lsi_model

def rp(dataframe, num_topics=300):
    """Returns an RP model for documents stored in a DataFrame.

    Precomputed models are read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        The DataFrame containing the documents to process.
    num_topics : int (default is 300)
        The number of topics to train the RP model with.

    Returns
    -------
    model : Gensim RpModel
        RP model for documents stored in the DataFrame.
    """
    filename = 'caches/models/rp.model'

    if not os.path.isfile(filename):
        dictionary = dictionary_corpus(dataframe)
        bow = bow_corpus(dataframe)
        tfidf_model = tfidf(dataframe)
        tfidf_corpus = tfidf_model[bow]
        rp_model = RpModel(tfidf_corpus, id2word=dictionary, num_topics=num_topics)
        rp_model.save(filename)
    else:
        rp_model = RpModel.load(filename)

    return rp_model

def lda(dataframe, num_topics=100):
    """Returns an LDA model for documents stored in a DataFrame.

    Precomputed models are read from file if previously cached, or generated then cached otherwise.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        The DataFrame containing the documents to process.
    num_topics : int (default is 300)
        The number of topics to train the LDA model with.

    Returns
    -------
    model : Gensim LdaMulticore
        LDA model for documents stored in the DataFrame.
    """
    filename = 'caches/models/lda.model'

    if not os.path.isfile(filename):
        dictionary = dictionary_corpus(dataframe)
        bow = bow_corpus(dataframe)
        lda_model = LdaModel(bow, id2word=dictionary, num_topics=num_topics, passes=20)
        lda_model.save(filename)
    else:
        lda_model = LdaModel.load(filename)

    return lda_model

_word2vec_model = None
def word2vec():
    """Returns a word2vec model pretrained on the Google News dataset.

    A single instance of the model is lazily loaded.

    Returns
    -------
    model : Gensim Word2Vec
        word2vec model pretrained on the Google News dataset.
    """
    global _word2vec_model

    if not _word2vec_model:
        filename = 'pretrained_models/GoogleNews-vectors-negative300.bin.gz'
        if not os.path.isfile(filename):
            wget.download('https://s3.ca-central-1.amazonaws.com/fnc-1/GoogleNews-vectors-negative300.bin.gz', out='pretrained_models')
        _word2vec_model = KeyedVectors.load_word2vec_format('pretrained_models/GoogleNews-vectors-negative300.bin.gz', binary=True)
        _word2vec_model.init_sims(replace=True)

    return _word2vec_model

def doc2vec(dataframe, size=100):
    """Returns a doc2vec model for documents stored in a DataFrame.

    Precomputed models are read from file previously cached, or generated then cached otherwise.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        The DataFrame containing the documents to process.
    size : int (default is 100)
        Dimensionality of the generated document vectors.

    Returns
    -------
    model : Gensim Doc2Vec
        doc2vec model for documents stored in the DataFrame.
    """
    filename = 'caches/models/doc2vec.model'

    if not os.path.isfile(filename):
        corpus = text_corpus(dataframe)
        tagged_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
        doc2vec_model = Doc2Vec(tagged_documents, size=size, window=10, min_count=2, iter=20, workers=multiprocessing.cpu_count())
        doc2vec_model.delete_temporary_training_data(keep_doctags_vectors=False)
        doc2vec_model.save(filename)
    else:
        doc2vec_model = Doc2Vec.load(filename)

    return doc2vec_model