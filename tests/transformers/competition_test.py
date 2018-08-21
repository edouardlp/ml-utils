import pytest

from mlutils.transformers.competition import WholeDatasetFit

from unittest.mock import Mock

def test_fit_calls_transformer():

    X = [[1,2],[3,4]]
    y = [1,2]
    whole_dataset = [1,3,4]

    transformer = Mock()
    wrapper = WholeDatasetFit(transformer=transformer, whole_dataset=whole_dataset)
    _test_fit_calls_transformer(X, y, whole_dataset, wrapper, transformer)

def _test_fit_calls_transformer(X, y, whole_dataset, wrapper, transformer):
    mock = Mock(return_value=transformer)
    transformer.fit = mock
    wrapper.fit(X, y)
    mock.assert_called_once_with(whole_dataset,None)
    pass

def test_transform_calls_transformer():

    X = [[1,2],[3,4]]
    whole_dataset = [1,3,4]

    transformer = Mock()
    wrapper = WholeDatasetFit(transformer=transformer, whole_dataset=whole_dataset)
    _test_transform_calls_transformer(X, wrapper, transformer)

def _test_transform_calls_transformer(X, wrapper, transformer):
    mock = Mock(return_value=transformer)
    transformer.transform = mock
    wrapper.transform(X)
    mock.assert_called_once_with(X)
    pass

if __name__ == '__main__':
    pytest.main([__file__])
