import pytest
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier

from mlutils.transformers import catboost

from unittest.mock import Mock

def test_fit_calls_estimator():

    X = [[1,2],[3,4]]
    y = [1,2]
    cat_features = [1,3,4]

    estimator = CatBoostRegressor()
    wrapper = catboost.Regressor(estimator=estimator, categorical_features=cat_features)
    _test_fit_calls_estimator(X, y, cat_features, wrapper, estimator)

    estimator = CatBoostClassifier()
    wrapper = catboost.Classifier(estimator=estimator, categorical_features=cat_features)
    _test_fit_calls_estimator(X, y, cat_features, wrapper, estimator)

def _test_fit_calls_estimator(X, y, categorical_features, wrapper, estimator):
    mock = Mock(return_value=estimator)
    estimator.fit = mock
    wrapper.fit(X, y)
    mock.assert_called_once_with(X,y,cat_features=categorical_features)
    pass

def test_predict_calls_estimator():

    X = [[1,2],[3,4]]
    cat_features = [1,3,4]

    estimator = CatBoostRegressor()
    wrapper = catboost.Regressor(estimator=estimator, categorical_features=cat_features)
    _test_predict_calls_estimator(X, wrapper, estimator)

    estimator = CatBoostClassifier()
    wrapper = catboost.Classifier(estimator=estimator, categorical_features=cat_features)
    _test_predict_calls_estimator(X, wrapper, estimator)

def _test_predict_calls_estimator(X, wrapper, estimator):
    mock = Mock(return_value=estimator)
    estimator.predict = mock
    wrapper.predict(X)
    mock.assert_called_once_with(X)
    pass

if __name__ == '__main__':
    pytest.main([__file__])