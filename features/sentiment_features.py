import numpy as np
import pandas as pd
from textacy.doc import Doc
from textacy.lexicon_methods import emotional_valence
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.notebook import in_notebook
from .feature import Feature

if in_notebook():
    from tqdm import tqdm_notebook
    tqdm = tqdm_notebook()
else:
    from tqdm import tqdm

class PolarityScorer(Feature):
    """VADER polarity scores.

    VADER expects unpreprocessed text.
    """
    @staticmethod
    def polarity_scores(dataframe):
        analyzer = SentimentIntensityAnalyzer()

        def polarity(text):
            doc = Doc(text, lang='en')
            sentences = [span.text for span in doc.sents]
            scores = [analyzer.polarity_scores(sentence) for sentence in sentences]
            np_scores = [np.array([score['neg'], score['neu'], score['pos'], score['compound']]) for score in scores]
            return np.mean(np.stack(np_scores), axis=0)

        tqdm.pandas(desc='Headline -> Polarity')
        headline_polarities = np.stack(dataframe['Headline'].progress_apply(polarity).values)
        headline_neg = headline_polarities[:,0]
        headline_neu = headline_polarities[:,1]
        headline_pos = headline_polarities[:,2]
        headline_compound = headline_polarities[:,3]

        tqdm.pandas(desc='articleBody -> Polarity')
        body_polarities = np.stack(dataframe['articleBody'].progress_apply(polarity).values)
        body_neg = body_polarities[:,0]
        body_neu = body_polarities[:,1]
        body_pos = body_polarities[:,2]
        body_compound = body_polarities[:,3]

        return pd.DataFrame({
            'Headline Polarity Neg': headline_neg,
            'Headline Polarity Neu': headline_neu,
            'Headline Polarity Pos': headline_pos,
            'Headline Polarity Compound': headline_compound,
            'articleBody Polarity Neg': body_neg,
            'articleBody Polarity Neu': body_neu,
            'articleBody Polarity Pos': body_pos,
            'articleBody Polarity Compound': body_compound
        }, index=dataframe.index)

    @classmethod
    def get_feature_generator(cls):
        return cls.polarity_scores

class EmotionScorer(Feature):
    """Depeche Mood emotional valence scores."""
    @staticmethod
    def emotional_valence(text):
        doc = Doc(text, lang='en')
        scores = emotional_valence(doc.tokens, dm_data_dir='pretrained_models')
        return np.array([scores['AFRAID'], scores['AMUSED'], scores['ANGRY'], scores['ANNOYED'], scores['DONT_CARE'], scores['HAPPY'], scores['INSPIRED'], scores['SAD']])

    @classmethod
    def emotion_scores(cls, dataframe):
        tqdm.pandas(desc='Headline -> Emotion')
        headline_emotions = np.stack(dataframe['Headline'].progress_apply(cls.emotional_valence).values)
        headline_afraid = headline_emotions[:,0]
        headline_amused = headline_emotions[:,1]
        headline_angry = headline_emotions[:,2]
        headline_annoyed = headline_emotions[:,3]
        headline_dont_care = headline_emotions[:,4]
        headline_happy = headline_emotions[:,5]
        headline_inspired = headline_emotions[:,6]
        headline_sad = headline_emotions[:,7]

        tqdm.pandas(desc='articleBody -> Emotion')
        body_emotions = np.stack(dataframe['articleBody'].progress_apply(cls.emotional_valence).values)
        body_afraid = body_emotions[:,0]
        body_amused = body_emotions[:,1]
        body_angry = body_emotions[:,2]
        body_annoyed = body_emotions[:,3]
        body_dont_care = body_emotions[:,4]
        body_happy = body_emotions[:,5]
        body_inspired = body_emotions[:,6]
        body_sad = body_emotions[:,7]

        return pd.DataFrame({
            'Headline Emotion Afraid': headline_afraid,
            'Headline Emotion Amused': headline_amused,
            'Headline Emotion Angry': headline_angry,
            'Headline Emotion Annoyed': headline_annoyed,
            'Headline Emotion Dont\'t Care': headline_dont_care,
            'Headline Emotion Happy': headline_happy,
            'Headline Emotion Inspired': headline_inspired,
            'Headline Emotion Sad': headline_sad,
            'articleBody Emotion Afraid': body_afraid,
            'articleBody Emotion Amused': body_amused,
            'articleBody Emotion Angry': body_angry,
            'articleBody Emotion Annoyed': body_annoyed,
            'articleBody Emotion Dont\'t Care': body_dont_care,
            'articleBody Emotion Happy': body_happy,
            'articleBody Emotion Inspired': body_inspired,
            'articleBody Emotion Sad': body_sad,
        }, index=dataframe.index)

    @classmethod
    def get_feature_generator(cls):
        return cls.emotion_scores