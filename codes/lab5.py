from util import progressbar
import pandas as pd
import numpy as np
from math import isinf


if __name__ == "__main__":
    LABEL = "Rented Bike Count"
    # get the data
    filename = "SeoulBikeData.csv"
    original_data = pd.read_csv("dataset/" + filename)
    model_name = "model_for_" + filename.split(".")[0]

    new_train = False

    # preprocessing
    original_data.drop("Date", axis=1, inplace=True)

    cols = ["Rented Bike Count", "Hour", "Visibility (10m)", "Humidity(%)"]
    for col in cols:
        original_data[col] = original_data[col].astype("float64")

    # original_data.info()
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    attributes_cat = ['Seasons', 'Holiday', 'Functioning Day']
    attributes_num = ['Hour', 'Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                      'Dew point temperature(C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']

    full_transformer = ColumnTransformer([
        ("categorical", OneHotEncoder(), attributes_cat),
        ("numerical", StandardScaler(), attributes_num),
    ])

    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(original_data, test_size=0.15, random_state=42)

    #
    train_set_vars = train_set.drop(LABEL, axis=1)
    train_set_labels = train_set[LABEL]

    test_set_vars = test_set.drop(LABEL, axis=1)
    test_set_labels = test_set[LABEL]

    # apply standardisation and one hot coder to the data
    train_set_vars_prepared = full_transformer.fit_transform(train_set_vars)
    test_set_vars_prepared = full_transformer.transform(test_set_vars)

    # transform numpy.ndarray to torch.tensor
    import torch

    # For the input variables
    train_torch = torch.from_numpy(train_set_vars_prepared)
    # Considering offsets, add 1s to the input variables
    # print(train_torch.shape)
    train_torch = torch.cat(
        (torch.ones((train_torch.shape[0], 1)), train_torch), dim=1
    )
    # print(train_torch.shape)
    # print(train_torch[:, 0])

    test_torch = torch.from_numpy(test_set_vars_prepared)
    # Considering offsets, add 1s to the input variables
    test_torch = torch.cat(
        (torch.ones((test_torch.shape[0], 1)), test_torch), dim=1
    )

    # for the labels
    train_torch_labels = torch.from_numpy(train_set_labels.to_numpy())
    test_torch_labels = torch.from_numpy(test_set_labels.to_numpy())

    # Done: Transformation into torch.tensor completed!

    w = torch.load("models/" + model_name)
    if isinstance(w, torch.Tensor):
        print("Model found. \n")
        new_train = False

    # HACK
    # new_train = True

    # the model prediction function
    def model_prediction(X, w):
        # return X @ w
        return torch.exp(X @ w)

    def loss_function_mse(y_true, y_approx):
        return torch.pow((y_true - y_approx), 2).mean()

    if new_train:
        print("New training process begins.", end="\n\n")
        # prepare the tensor for calculate the gradient

        # w: it is this tensor that need to calculate gradient
        # HACK: w can be zeros and max_loops can be very large.
        # w = torch.randn((train_torch.shape[1], 1), dtype=torch.float64, requires_grad=True)
        w = torch.zeros(
            (train_torch.shape[1], 1), dtype=torch.float64, requires_grad=True)
        max_loops = 5

        eta = 0.000001

        # begin training
        # w{k+1} = w{k} - η * ∂E(w)/∂w

        # for niter in range(max_loops):
        for niter in progressbar(range(max_loops), "Training:", 20):
            y_approx = model_prediction(train_torch, w)
            loss = loss_function_mse(train_torch_labels, y_approx)

            if isinf(loss):
                print(f"At {niter}th loop, loss becomes infinity.\nExiting.")
                exit(1)

            loss.backward()

            # TODO: why is it necessary?
            with torch.no_grad():
                w -= eta * w.grad

            w.grad.zero_()

            # generate output
            if niter % 5 == 0:
                print(f"Iteration {niter+1}: loss = {loss:.8f}")

        # find the w that minimize the loss function
        # print(w)
        torch.save(w, "models/" + model_name)
        print()
        print(f"After {max_loops} loops:")
        print("Model saved at", "models/" + model_name, end="\n\n")

    # RMSD
    w = w.detach()

    # calculate the RMSD for train set
    train_torch_predictions = train_torch @ w
    RMSD = np.sqrt(loss_function_mse(
        train_torch_labels, train_torch_predictions))

    print(
        f"For train set:\nThe root-mean-square deviation = {RMSD:.4f}", end="\n\n")

    # calculate the RMSD for test set
    test_torch_predictions = test_torch @ w
    RMSD = np.sqrt(loss_function_mse(
        test_torch_labels, test_torch_predictions))

    print(
        f"For test set:\nThe root-mean-square deviation = {RMSD:.4f}", end="\n\n")
