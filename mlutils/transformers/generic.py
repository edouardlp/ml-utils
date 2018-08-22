from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

class SerialTransformer(BaseEstimator, TransformerMixin):

    """

    SerialTransformer chains multiples transformers into a single transformer.

    This is useful for cases where only one transformer is supported

    """

    def __init__(self, transformers):
        self.transformers = transformers

    def transform(self, X):
        result_X = X
        for transformer in self.transformers[:-1]:
            result_X = transformer.transform(result_X)
        last = self.transformers[-1]
        return last.transform(result_X)

    def fit(self, X, y):
        result_X = X
        for transformer in self.transformers[:-1]:
            transformer.fit(result_X, y)
            result_X = transformer.transform(result_X)
        last = self.transformers[-1]
        return last.fit(result_X, y)


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):

    """

    MultiColumnLabelEncoder applies a label encoder to each of the specified column indices

    This is useful for pipelines where there are multiple columns to encode with distinct vocabularies.

    """
    label_encoders = {}

    def __init__(self, columns):
        self.columns = columns

    def transform(self, X):
        result = X.copy()
        for feature in self.columns:
            label_encoder = self.label_encoders[feature]
            label_encoded = label_encoder.transform(X[:,feature])
            result[:, feature] = label_encoded
        return result

    def fit(self, X, y):
        for feature in self.columns:
            label_encoder = LabelEncoder()
            label_encoder.fit(X[:,feature])
            self.label_encoders[feature] = label_encoder
        return self

class SelectiveColumnTransformer(BaseEstimator, TransformerMixin):

    """

    SelectiveColumnTransformer applies a transformer only to the specified column indices.

    This is useful for pipelines.

    """

    transformers_by_column = {}

    def __init__(self, transformer, columns):
        self.transformer = transformer
        self.columns = columns

    def transform(self, X):
        result = X.copy()
        for column in self.columns:
            this_transformer = self.transformers_by_column[column]
            transformed = this_transformer.transform(X[:,column].reshape(-1,1))
            result[:, column] = transformed.flatten()
        return result

    def fit(self, X, y):
        for column in self.columns:
            this_transformer = clone(self.transformer)
            this_transformer.fit(X[:,column].reshape(-1,1), y)
            self.transformers_by_column[column] = this_transformer
        return self
