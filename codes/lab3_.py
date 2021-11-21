import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
LABEL = "Rented Bike Count"

attributes_categorical = ['Seasons', 'Holiday', 'Functioning Day']
attributes_numerical = ['Hour', 'Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                        'Dew point temperature(C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']


def random_forest():
    

    # Get the data
    original_data = pd.read_csv(
        'dataset/SeoulBikeData.csv', encoding='unicode_escape')

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
        ("numberical", StandardScaler(), attributes_numerical),
        ("categorical", OneHotEncoder(), attributes_categorical)
    ])

    train_subset_attributes_prepared = full_transformer.fit_transform(
        train_subset_attributes)
    val_set_attributes_prepared = full_transformer.transform(val_set_attributes)

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

    # prepare for the GridSearchCV to find the best parameter for estimator RandomForestRegressor
    # two adjustable parameters
    # 1. n_estimators: int, default=100 - The number of trees in the forest.
    # 2. max_samples

    max_samples_options = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]
    n_estimators_options = [10, 15, 19, 20, 21, 22, 25, 30, 200]

    param_grid = dict(n_estimators=n_estimators_options, max_samples=max_samples_options)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    grid_search_cv = GridSearchCV(
        estimator = RandomForestRegressor(),
        param_grid = param_grid,
        scoring="neg_mean_squared_error",
        cv=ps
    )
    # grid_search_cv.fit(whole_train_set_attributes, whole_train_set_labels)
    # print(grid_search_cv.best_params_)
    # {'max_depth': 14, 'n_estimators': 25}

    # use the best parameter to train model with train_subset_attributes_prepared and train_subset_labels
    rfr = RandomForestRegressor(
        # n_estimators = grid_search_cv.best_params_["n_estimators"],
        # max_samples = grid_search_cv.best_params_["max_samples"],
        n_estimators = 200,
        max_samples = 3000
    )
    rfr.fit(train_subset_attributes_prepared, train_subset_labels)


    # calculate the RMSD: root-mean-square deviation

    # for validate set:
    predictions_val_set = rfr.predict(val_set_attributes_prepared)
    from sklearn.metrics import mean_squared_error
    RMSD = np.sqrt(mean_squared_error(val_set_labels, predictions_val_set))
    print("For validate set:\nRMSD = %.4f" % RMSD)
    # 236.6382

    # for test
    test_set_attributes = test_set.drop(LABEL, axis=1)
    test_set_labels = test_set[LABEL]
    test_set_attributes_prepared = full_transformer.transform(
        test_set_attributes)
    predictions_test_set = rfr.predict(
        test_set_attributes_prepared)

    RMSD1 = np.sqrt(mean_squared_error(test_set_labels, predictions_test_set))
    print("For test set:\nRMSD = %.4f" % RMSD1)


if __name__ == "__main__":
    random_forest()
