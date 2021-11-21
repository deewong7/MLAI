import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
LABEL = "Rented Bike Count"

attributes_categorical = ['Seasons', 'Holiday', 'Functioning Day']
attributes_numerical = ['Hour', 'Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                        'Dew point temperature(C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']


def decision_tree_for_regression():

    original_data = pd.read_csv(
        'dataset/SeoulBikeData.csv', encoding='unicode_escape')
    
    original_data.drop("Date", axis=1, inplace=True)

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
        ("categorical", OneHotEncoder(), attributes_categorical),],
        # HACK:
        remainder="passthrough"
    )
    """
    remainder{‘drop’, ‘passthrough’}, default=’drop’

    By default, only the specified columns in transformers are transformed and combined in the output,
    and the non-specified columns are dropped.

    By specifying remainder='passthrough', all remaining columns
    that were not specified in transformers will be automatically passed through.
    This subset of columns is concatenated with the output of the transformers.
    
    By setting remainder to be an estimator, the remaining non-specified columns will use the remainder estimator.
    The estimator must support fit and transform.
    Note that using this feature requires that the DataFrame columns input at fit and transform have identical order.
    """


    train_subset_attributes_prepared = full_transformer.fit_transform(
        train_subset_attributes)
    val_set_attributes_prepared = full_transformer.transform(
        val_set_attributes)

    # to perform a GridSearchCV on whole train set
    test_fold = np.zeros((np.shape(train_set)[0], 1))
    test_fold[0:np.shape(train_subset)[0]] = -1

    # use PredefinedSplit as a cross validate method
    from sklearn.model_selection import PredefinedSplit
    ps = PredefinedSplit(test_fold)  # one split

    whole_train_set_attributes = np.vstack(
        (train_subset_attributes_prepared, val_set_attributes_prepared))
    whole_train_set_labels = np.hstack((train_subset_labels, val_set_labels))

    # prepare for the GridSearchCV to find the best parameter
    max_depth_options = [3, 5, 10, 15]
    param_grid = dict(max_depth=max_depth_options)

    from sklearn import tree
    from sklearn.model_selection import GridSearchCV
    grid_regression = GridSearchCV(tree.DecisionTreeRegressor(
    ), param_grid=param_grid, cv=ps, scoring='neg_mean_squared_error')
    grid_regression.fit(whole_train_set_attributes, whole_train_set_labels)
    # print(grid_regression.best_params_)  # max_depth = 10

    # use the best parameter to train the model
    decision_tree_regressor = tree.DecisionTreeRegressor(
        max_depth=grid_regression.best_params_["max_depth"], random_state=42)
    decision_tree_regressor.fit(
        train_subset_attributes_prepared, train_subset_labels)
    predictions_val_set = decision_tree_regressor.predict(
        val_set_attributes_prepared)

    # calculate the RMSD: root-mean-square deviation
    from sklearn.metrics import mean_squared_error
    RMSD = np.sqrt(mean_squared_error(val_set_labels, predictions_val_set))
    print("For validate set:\nRMSD = %.4f" % RMSD)

    test_set_attributes = test_set.drop(LABEL, axis=1)
    test_set_labels = test_set[LABEL]
    test_set_attributes_prepared = full_transformer.transform(
        test_set_attributes)
    predictions_test_set = decision_tree_regressor.predict(
        test_set_attributes_prepared)

    RMSD1 = np.sqrt(mean_squared_error(test_set_labels, predictions_test_set))
    print("For test set:\nRMSD = %.4f" % RMSD1)


if __name__ == "__main__":
    decision_tree_for_regression()
