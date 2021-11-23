import numpy as np


def analyse(item, with_d=False):
    print()
    print("Type:", type(item))
    print("Item:", item)

    try:
        print("Leng:", len(item))
    except Exception:
        print("function len() is not applicable to this object.")

    if with_d:
        print("Dimension:", item.shape)

    print()


def polynomial(x, num_basis=4, data_limits=[-1., 1.]):
    X = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        # NOTE: where i:i+1 is a key to ensure the X[:, i:i+1].shape is equal to x**i
        # x**i.shape = (27, 1)
        X[:, i:i+1] = x**i
    return X
