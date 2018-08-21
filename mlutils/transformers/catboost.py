from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin

class Base(BaseEstimator):

    """

    Wrapper around Catboost to support specifying categorical features during init.

    This is useful for frameworks relying on the scikit-learn estimator API.

    """
    def __init__(self,
                 estimator,
                 categorical_features):
        self.estimator = estimator
        self.categorical_features = categorical_features

    def fit(self, X, y):
        return self.estimator.fit(X, y, cat_features=self.categorical_features)

    def predict(self, X):
        return self.estimator.predict(X)

class Regressor(Base, RegressorMixin):

    """

    Wrapper around Catboost Regressor to support specifying categorical features during init.

    This is useful for frameworks relying on the scikit-learn estimator API.

    """
    pass

class Classifier(Base, ClassifierMixin):

    """

    Wrapper around Catboost Classifier to support specifying categorical features during init.

    This is useful for frameworks relying on the scikit-learn estimator API.

    """
    pass