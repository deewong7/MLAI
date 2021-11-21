import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
LABEL = "Rented Bike Count"

attributes_categorical = ['Seasons', 'Holiday', 'Functioning Day']
attributes_numerical = ['Hour', 'Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                        'Dew point temperature(C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']


def gradient_boosting_regressor():

    # Get the data
    original_data = pd.read_csv(
        'dataset/SeoulBikeData.csv', encoding='unicode_escape')

    original_data.drop("Date", axis=1, inplace=True)

    # Data pre-processing
    for col in ['Rented Bike Count', 'Hour', 'Humidity(%)', 'Visibility (10m)']:
        original_data[col] = original_data[col].astype("float64")

    train_set, test_set = train_test_split(
        original_data, test_size=0.15, random_state=42)

    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    train_subset, val_set = train_test_split(
        train_set, test_size=0.15, random_state=42)

    train_subset_attributes = train_subset.drop(LABEL, axis=1)
    train_subset_labels = train_subset[LABEL]

    val_set_attributes = val_set.drop(LABEL, axis=1)
    val_set_labels = val_set[LABEL]

    full_transformer = ColumnTransformer([
        # ("numberical", StandardScaler(), attributes_numerical),
        ("categorical", OneHotEncoder(), attributes_categorical)
    ], remainder="passthrough")

    train_subset_attributes_prepared = full_transformer.fit_transform(
        train_subset_attributes)
    val_set_attributes_prepared = full_transformer.transform(
        val_set_attributes)

    # to perform a GridSearchCV on whole train set
    test_fold = np.zeros((np.shape(train_set)[0], 1))
    test_fold[0:np.shape(train_subset)[0]] = -1

    # use PredefinedSplit as a cross validate method
    from sklearn.model_selection import PredefinedSplit
    ps = PredefinedSplit(test_fold)  # about how to split the data:
    # one split with the sub_train_set as train set,
    # with the val_set as test set.

    whole_train_set_attributes = np.vstack(
        (train_subset_attributes_prepared, val_set_attributes_prepared))
    whole_train_set_labels = np.hstack((train_subset_labels, val_set_labels))

    # prepare for the GridSearchCV to find the best parameter for estimator GradientBoostingRegressor
    # two adjustable parameters
    # 1. n_estimators
    # 2. max_features

    max_features_options = list(range(13))[1:]
    # n_estimators_options = [10, 30, 200, 300, 400, 500, 600]
    n_estimators_options = [600, 650, 680, 700, 750, 900]

    param_grid = dict(n_estimators=n_estimators_options,
                      max_features=max_features_options)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GridSearchCV
    grid_search_cv = GridSearchCV(
        estimator=GradientBoostingRegressor(),
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=ps
    )
    grid_search_cv.fit(whole_train_set_attributes, whole_train_set_labels)
    print(grid_search_cv.best_params_)
    # {'max_features': 14, 'n_estimators': 25}

    # use the best parameter to train model with train_subset_attributes_prepared and train_subset_labels
    gbr = GradientBoostingRegressor(
        n_estimators=grid_search_cv.best_params_["n_estimators"],
        max_features=grid_search_cv.best_params_["max_features"],
        # n_estimators = 600,
        # max_featuress = 8
    )
    gbr.fit(train_subset_attributes_prepared, train_subset_labels)

    # calculate the RMSD: root-mean-square deviation

    # for validate set:
    predictions_val_set = gbr.predict(val_set_attributes_prepared)
    from sklearn.metrics import mean_squared_error
    RMSD = np.sqrt(mean_squared_error(val_set_labels, predictions_val_set))
    print("For validate set:\nRMSD = %.4f" % RMSD)
    # 236.6382

    # for test
    test_set_attributes = test_set.drop(LABEL, axis=1)
    test_set_labels = test_set[LABEL]
    test_set_attributes_prepared = full_transformer.transform(
        test_set_attributes)
    predictions_test_set = gbr.predict(
        test_set_attributes_prepared)

    RMSD1 = np.sqrt(mean_squared_error(test_set_labels, predictions_test_set))
    print("For test set:\nRMSD = %.4f" % RMSD1)


if __name__ == "__main__":
    gradient_boosting_regressor()

    """
    {'max_features': 5, 'n_estimators': 900}

    For validate set:
    RMSD = 227.0208

    For test set:
    RMSD = 236.6605

    """
