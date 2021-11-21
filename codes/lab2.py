"""
The purpose of this file is simple, to re-practice/apply the concepts in the Labs 2 (An end-to-end project in Machine Learning),
to train a model to predict bike retals.

The data is from UCI Machine Learning Repository, link below:
https://archive.ics.uci.edu/ml/index.php

Author: Di Wang (dwang84@sheffield.ac.uk)
Date: 2021-11-19 15:07:29

Keywords: scikit-learn, standardisation and normalisation
"""


"""
Machine Learning Project Checklist
1. Frame the problem and look at the big picture.
2. Get the data.
3. Explore the data to get insights.
4. Prepare the data to better expose the underlying data patterns.
5. Explore many different models and shortlist the best ones.
6. Fine-tune your models and combine them into a soluttion.
7. Present your solution.
"""

# typings

# 1. Get the Data


from pandas.core.frame import DataFrame
from pandas.core.frame import Series
from typing import Tuple
from numpy import ndarray
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
filename = "dataset/" + url.split("/")[-1]

LABELS = "Rented Bike Count"


def analyse(item):
    print()
    print(type(item))
    print(item)
    print()


def get_the_data():
    global filename
    import urllib.request as request
    request.urlretrieve(url, filename)


def explore_the_data() -> Tuple[DataFrame, DataFrame, DataFrame]:
    global filename
    import pandas as pd
    original_dataset = pd.read_csv(filename)
    # print(original_dataset.describe())
    # print(original_dataset.sample(2))
    # print(original_dataset.columns)

    # there is an inplace parameter
    # res = original_dataset.drop("Date", axis=1)
    original_dataset.drop("Date", axis=1, inplace=True)
    # original_dataset.info() # returns NoneType

    for col in ["Rented Bike Count", "Hour", "Humidity(%)", "Visibility (10m)"]:
        original_dataset[col] = original_dataset[col].astype("float64")

    # original_dataset.info() # returns NoneType

    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(
        original_dataset, test_size=0.15, random_state=42)
    # type(train_set) # <class 'pandas.core.frame.DataFrame'>
    sub_train_set, validate_set = train_test_split(train_set, test_size=0.15)
    return (sub_train_set, validate_set, test_set)

    import matplotlib.pyplot as plt
    # train_set.hist(bins=50, figsize=(20, 15))
    # plt.show()

    from pandas.plotting import scatter_matrix
    attributes = ["Rented Bike Count", "Hour",
                  "Temperature(C)", "Humidity(%)", "Wind speed (m/s)"]
    # fig = scatter_matrix(train_set[attributes], figsize=(15, 10))
    # plt.show()

    correlation_coefficient_matrix = train_set.corr()
    analyse(correlation_coefficient_matrix["Rented Bike Count"].sort_values(
        ascending=False))


def prepare_data(train_set: DataFrame) -> Tuple[ndarray, Series, ColumnTransformer]:
    attributes_cat = ['Seasons', 'Holiday', 'Functioning Day']
    attributes_num = ['Hour', 'Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                      'Dew point temperature(C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']

    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer

    full_transformer = ColumnTransformer([
        ("Numberic values", StandardScaler(), attributes_num),
        ("Categorical values", OneHotEncoder(), attributes_cat)
    ])

    train_set_input_vars = train_set.drop(LABELS, axis=1)
    train_set_labels = train_set[LABELS]

    train_set_input_vars_prepared = full_transformer.fit_transform(
        train_set_input_vars)
    # analyse(train_set_input_vars_prepared)

    return train_set_input_vars_prepared, train_set_labels, full_transformer


def train(train_set_input_vars_prepared, train_set_labels) -> LinearRegression:
    from sklearn.linear_model import LinearRegression
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(
        train_set_input_vars_prepared, train_set_labels)

    return linear_regression_model


def validate(transformer: ColumnTransformer, validate_set: DataFrame, model: LinearRegression):

    validate_set_input_vars = validate_set.drop(LABELS, axis=1)
    validate_set_labels = validate_set[LABELS]

    validate_set_input_vars_prepared = transformer.transform(
        validate_set_input_vars)
    predictions = model.predict(validate_set_input_vars_prepared)

    from sklearn.metrics import mean_squared_error
    from math import sqrt
    RMSD = sqrt(mean_squared_error(validate_set_labels, predictions))
    print("For validate set: root-mean-square deviation = ", RMSD)

    return RMSD


def test(test_set: DataFrame, transformer: ColumnTransformer, model: LinearRegression):

    test_set_input_vars = test_set.drop(LABELS, axis=1)
    test_set_labels = test_set[LABELS]

    test_set_input_vars_prepared = transformer.transform(test_set_input_vars)

    predictions = model.predict(test_set_input_vars_prepared)

    from sklearn.metrics import mean_squared_error
    from math import sqrt
    RMSD = sqrt(mean_squared_error(test_set_labels, predictions))
    print("For test set: root-mean-square deviation =", RMSD)

    return(RMSD)


if __name__ == "__main__":
    # get_the_data()
    train_set, validate_set, test_set = explore_the_data()

    train_set_input_vars_prepared, train_set_labels, transformer = prepare_data(
        train_set)

    linear_regression_model = train(
        train_set_input_vars_prepared, train_set_labels)

    # validate(transformer, validate_set, linear_regression_model)
    # For validate: Root-mean-square deviation = 432.2275924342351

    # fine-tune ...
    # fine-tune ...
    # fine-tune ...

    test(test_set, transformer, linear_regression_model)
    # For test: root-mean-square deviation = 432.91609335164463
    # slightly higher
