import numpy as np
import pandas as pd

from joblib import dump, load

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

import scipy.stats as stats

features = ["year", "km_driven", "seats", "max_power", "fuel", "transmission"]
numeric_cols = ['year', 'km_driven', 'seats', 'max_power']
categorical_cols = ['fuel', 'transmission']
features = ["year", "km_driven", "seats", "max_power", "fuel", "transmission"]
drop_cols = ["owner", "year", "seller_type", "mileage", "engine"]

numeric_cols = [0, 1, 4, 9]
categorical_cols = [5, 6]
last_col = -1
replace_fuel = {'Petrol': 1, 'Diesel': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5}
replace_transmission = {'Manual': 1, 'Automatic': 2}


# Custom transformer to calculate the antiquity of a car
class AntiquityCalculatorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_col = 2020 - X[:, 0]
        return np.c_[new_col, X]

# Replacer to transform all the categorical values in to numerical ones
def replace_values(arr, replace_dict):
    for k, v in replace_dict.items():
        arr[arr == k] = v
    return arr

# Lambda function necesarry of the transformer won't work
def my_lambda(x):
    return x.values

# Numeric transformer: impute all the passed on numerical columns so that all null values are replaced with the median of each column
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

# Categorical transformer: impute all the passed on categorical columns so that all null values are replaced with the mode of each column
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Force the dataframe to become an ndarray so that all transformations are easier
data_pipeline = Pipeline([
    ('to_numpy', FunctionTransformer(my_lambda, validate=False))
])

# Replace all the elements in the "fuel" column so as to be able to run a regression
fuel_transformer = Pipeline(steps=[('replace_fuel', FunctionTransformer(replace_values, validate=False, kw_args={'replace_dict': replace_fuel}))])

# Replace all the elements in the "transmission" column so as to be able to run a regression
transmission_transformer = Pipeline(steps=[('replace_transmission', FunctionTransformer(replace_values, validate=False, kw_args={'replace_dict': replace_transmission}))])

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('numeric', numeric_transformer, numeric_cols),
    ('categorical', categorical_transformer, categorical_cols),
])

# Complete pipeline
pipeline = Pipeline([
    ('data_pipeline', data_pipeline),
    ('preprocessor', preprocessor),
    ('replace_fuel', fuel_transformer),
    ('replace_transmission', transmission_transformer),
    ('antiquity', AntiquityCalculatorTransformer()),
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])
