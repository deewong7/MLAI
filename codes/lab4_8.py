"""
About basis function in linear regression model where hte parameters are linear:

If we are doing a multivariate linear regression,
we get extra features that might help us predict our required response variable (or target value), y.

But what if we only have one input value? 
We can actually artificially generate more input values with basis functions.
"""


from util import analyse, polynomial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get the data
url = "https://raw.githubusercontent.com/maalvarezl/MLAI/master/Labs/datasets/olympic_marathon_men.csv"
original_data = pd.read_csv(url, header=None)
# NOTE: what if I do not reshape it?
# Only after reshape, x and y can become a vector/matrix and then transpose becomes applicable
# you do not need to the worry about the other dimension when passing -1 as the shape value
x = np.array(original_data.iloc[:, 0]).reshape(-1, 1)
y = np.array(original_data.iloc[:, 1]).reshape(-1, 1)


def linear_model(show: bool = False) -> np.ndarray:

    X = np.hstack((np.ones_like(x), x))

    w = np.linalg.solve(X.T @ X, X.T @ y)

    y_pred = X @ w

    if show:
        plt.plot(x, y_pred, 'r')
        plt.plot(x, y, 'g.')
        plt.show()

    return y_pred


def linear_model_with_basis_functions(num_basis: int = 5, show : bool = False) -> np.ndarray:

    X = polynomial(x, num_basis)
    W = np.linalg.solve(X.T @ X, X.T @ y)

    y_pred = X @ W

    if show:
        plt.plot(x, y, 'g.')
        plt.plot(x, y_pred, 'r-')
        plt.show()

    return y_pred


if __name__ == "__main__":
    y_pred = linear_model(True)
    # FIX: why error?
    y_pred_with_basis_function = linear_model_with_basis_functions(5, True)
    from sklearn.metrics import mean_squared_error

    RMSD = np.sqrt(mean_squared_error(y, y_pred))
    RMSD_with_basis_function = np.sqrt(mean_squared_error(y, y_pred_with_basis_function))

    print("RMSD for linear model only = %.5f." % RMSD)
    print("RMSD for linear model with basis function = %.5f." % RMSD_with_basis_function)
    print("The difference between them = %.5f." % np.abs(RMSD - RMSD_with_basis_function))
