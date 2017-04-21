import os
import pandas as pd
from abc import ABC, abstractmethod

class Feature(ABC):
    @classmethod
    @abstractmethod
    def get_feature_generator(cls):
        """Returns a function that generates features from a set of headlines and bodies.

        Returns
        -------
        feature_generator : function(dataframe)
            Generates features from a set of headlines and bodies.
        """
        pass

    @classmethod
    def transform(cls, dataframe, key=None):
        """Returns features, given a DataFrame containing a set of headlines and bodies.

        Features are read from file if previously cached, or generated then cached otherwise.

        Parameters
        ----------
        dataframe : Pandas DataFrame
            The dataframe containing a set of headlines and bodies.
        key : str or None
            Key used for caching.

        Returns
        -------
        features : array-like
            Features generated from the headlines and bodies. Can be multi-dimensional, depending on the feature generator.
        """
        feature_generator = cls.get_feature_generator()
        filename = 'caches/features/{}.pkl'.format(feature_generator.__name__) if key == None else 'caches/features/{}.{}.pkl'.format(key, feature_generator.__name__)

        if not os.path.isfile(filename):
            features = feature_generator(dataframe)
            features.to_pickle(filename)
        else:
            features = pd.read_pickle(filename)
            
        return features