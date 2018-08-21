import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class WholeDatasetFit(BaseEstimator, TransformerMixin):

    """

    WholeDatasetFit is a transformer that fits on a predefined dataset and ignores X passed in fit.

    This is useful in pipelines with cross validation used in a competition. This estimator is able to leak data from the test set for example.

    """

    def __init__(self, transformer, whole_dataset):
        self.transformer = transformer
        self.whole_dataset = whole_dataset

    def transform(self, X):
        return self.transformer.transform(X)

    def fit(self, X, y):
        return self.transformer.fit(self.whole_dataset, None)
