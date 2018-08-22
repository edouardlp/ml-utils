# ml-utils

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/edouardlp/ml-utils/blob/master/LICENSE)

## Overview

WORK IN PROGRESS

ml-utils is a simple collection of utilities that are convenient for prototyping ml models using scikit-learn.

------------------

## Installing

TBD

## Examples

### MultiColumnLabelEncoder

MultiColumnLabelEncoder applies a label encoder to each of the specified column indices

This is useful for pipelines where there are multiple columns to encode with distinct vocabularies.

Example of label encoding only certain columns

```python

from mlutils.transformers.generic import MultiColumnLabelEncoder

categorical_columns = ...
encoder = MultiColumnLabelEncoder(columns=categorical_columns)

```

### SelectiveColumnTransformer

SelectiveColumnTransformer applies a transformer only to the specified column indices.

This is useful for pipelines.

Example of scaling only numerical features

```python

from mlutils.transformers.generic import SelectiveColumnTransformer
from sklearn.preprocessing import StandardScaler

numerical_columns = None
scaler = StandardScaler()
scaler = SelectiveColumnTransformer(scaler, columns=numerical_columns)

```

### SerialTransformer

SerialTransformer chains multiples transformers into a single transformer.

This is useful for cases where only one transformer is supported

Example of building a one hot encoder that also encodes the labels

```python

from mlutils.transformers.generic import MultiColumnLabelEncoder
from mlutils.transformers.generic import SerialTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_columns = ...
transformers = [MultiColumnLabelEncoder(columns=categorical_columns),OneHotEncoder(categorical_features=categorical_columns)]
one_hot_encoder = SerialTransformer(transformers=transformers)

```


### WholeDatasetFit

WholeDatasetFit is a transformer that fits on a predefined dataset and ignores X passed in fit.

This is useful in pipelines with cross validation used in a competition. This estimator is able to leak data from the test set for example.

Example of fitting a Standard Scaler on the whole dataset :

```python

from sklearn.preprocessing import StandardScaler
from mlutils.transformers.competition import WholeDatasetFit

entire_dataset = ...
train_dataset = ...

scaler = StandardScaler()
entire_dataset_scaler = WholeDatasetFit(scaler, entire_dataset)
transformed_train_dataset = entire_dataset_scaler.fit_transform(train_dataset)
```
