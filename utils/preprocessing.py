import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from textacy.doc import Doc
from textacy.preprocess import preprocess_text

from .dataset import Dataset
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
        - replacing hone number strings with 'phone'
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
    filename = 'caches/preprocessing/' + key + '.h5'

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
    return encoder.transform(dataframe['Stance'].values)

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