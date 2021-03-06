import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from textacy.doc import Doc
from textacy.preprocess import preprocess_text

from .notebook import in_notebook

if in_notebook():
    from tqdm import tqdm_notebook
    tqdm = tqdm_notebook()
else:
    from tqdm import tqdm

def preprocess_text_string(text):
    """Preprocesses text for feature extraction.

    Preprocessing tasks are as follows:
        - whitespace normalization
        - fixing broken unicode via ftfy
        - converting text to lowercase
        - replacing url strings with 'url'
        - replacing phone number strings with 'phone'
        - replacing currency symbols with their standard 3-letter abbreviations
        - stripping punctuation
        - replacing contractions with their unshortened forms
        - lemmatizing words

    Parameters
    ----------
    text : str
        The input text to be preprocessed.

    Returns
    -------
    preprocessed : str
        The preprocessed output text.
    """
    text = preprocess_text(text, fix_unicode=True, lowercase=True, no_urls=True, no_phone_numbers=True, no_currency_symbols=True, no_punct=True, no_contractions=True)
    doc = Doc(text, lang='en')
    lemmatized_tokens = doc.to_terms_list(ngrams=1, named_entities=False, as_strings=True, normalize='lemma')
    return ' '.join(lemmatized_tokens)

def preprocess_dataframe(dataframe, key):
    """Preprocesses DataFrames for feature extraction. 

    Preprocesses the text of the "Headline" and "articleBody" columns.
    Preprocessed DataFrames are read from file if previously cached, or generated then cached otherwise.
    
    Parameters
    ----------
    dataframe : Pandas DataFrame
        The unprocessed DataFrame.
    key : str
        Key used for caching.

    Returns
    -------
    preprocessed : Pandas DataFrame
        The preprocessed DataFrame.
    """
    filename = 'caches/preprocessing/{}.h5'.format(key)

    if not os.path.isfile(filename):
        data = dataframe.copy()

        tqdm.pandas(desc='Preprocessing Headlines')
        data['Headline'] = data['Headline'].progress_apply(preprocess_text_string)
        
        tqdm.pandas(desc='Preprocessing articleBodies')
        data['articleBody'] = data['articleBody'].progress_apply(preprocess_text_string)
        
        data.to_hdf(filename, key, complevel=9, complib='blosc')
    else:
        data = pd.read_hdf(filename, key)

    return data

def extract_labels(dataframe):
    """Extracts stance labels from the DataFrame and converts to numeric label.
    
    Parameters
    ----------
    dataframe : Pandas DataFrame
        The DataFrame containing the stances to extract.

    Returns
    -------
    labels : Pandas DataFrame
        Contains the stances in numeric form.
    """
    encoder = LabelEncoder()
    encoder.fit(['unrelated', 'discuss', 'agree', 'disagree'])
    return pd.DataFrame({'Stance': encoder.transform(dataframe['Stance'].values)}, index=dataframe.index)

def decipher_labels(labels, index):
    """Converts numerically encoded labels back into textual stances.

    Parameters
    ----------
    labels : Numpy ndarray
        ndarray containing numerically encoded labels.

    Returns
    -------
    stances : Pandas DataFrame
        DataFrame containing textual stance labels.
    """
    encoder = LabelEncoder()
    encoder.fit(['unrelated', 'discuss', 'agree', 'disagree'])
    return pd.DataFrame({'Stance': encoder.inverse_transform(labels)}, index=index)

def oversample_minority_classes(features, labels):
    """Oversamples a dataset's minority classes using the SVM-SMOTE algorithm.

    Parameters
    ----------
    features : Pandas DataFrame
        DataFrame containing numeric features.
    labels : Pandas DataFrame
        DataFrame containing numeric stance labels.

    Returns
    -------
    sampled_features : numpy ndarray
        ndarray containing oversampled features.
    sampled_labels : numpy array
        array containing oversampled labels.
    """
    AGREE = 0
    DISAGREE = 1
    DISCUSS = 2
    UNRELATED = 3

    agree_labels = labels[labels['Stance'] == AGREE]
    disagree_labels = labels[labels['Stance'] == DISAGREE]
    discuss_labels = labels[labels['Stance'] == DISCUSS]
    unrelated_labels = labels[labels['Stance'] == UNRELATED]

    agree_features = features.loc[agree_labels.index]
    disagree_features = features.loc[disagree_labels.index]
    discuss_features = features.loc[discuss_labels.index]
    unrelated_features = features.loc[unrelated_labels.index]

    oversampler = SMOTE(kind='svm')

    print('Oversampling agree group...')
    agree_group_features = pd.concat([agree_features, unrelated_features])
    agree_group_labels = pd.concat([agree_labels, unrelated_labels])
    agree_group_features, agree_group_labels = oversampler.fit_sample(agree_group_features, agree_group_labels.values.ravel())
    agree_idx = np.where(agree_group_labels == AGREE)
    agree_features, agree_labels = agree_group_features[agree_idx], agree_group_labels[agree_idx]

    print('Oversampling disagree group...')
    disagree_group_features = pd.concat([disagree_features, unrelated_features])
    disagree_group_labels = pd.concat([disagree_labels, unrelated_labels])
    disagree_group_features, disagree_group_labels = oversampler.fit_sample(disagree_group_features, disagree_group_labels.values.ravel())
    disagree_idx = np.where(disagree_group_labels == DISAGREE)
    disagree_features, disagree_labels = disagree_group_features[disagree_idx], disagree_group_labels[disagree_idx]

    print('Oversampling discuss group...')
    discuss_group_features = pd.concat([discuss_features, unrelated_features])
    discuss_group_labels = pd.concat([discuss_labels, unrelated_labels])
    discuss_group_features, discuss_group_labels = oversampler.fit_sample(discuss_group_features, discuss_group_labels.values.ravel())
    discuss_idx = np.where(discuss_group_labels == DISCUSS)
    discuss_features, discuss_labels = discuss_group_features[discuss_idx], discuss_group_labels[discuss_idx]

    oversampled_features, oversampled_labels = np.concatenate((agree_features, disagree_features, discuss_features, unrelated_features.values)), np.concatenate((agree_labels, disagree_labels, discuss_labels, unrelated_labels.values.ravel()))
    return shuffle(oversampled_features, oversampled_labels)