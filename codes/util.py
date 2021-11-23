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


# progressbar
# author: iambr
# from stackoverflow: https://stackoverflow.com/a/34482761/9690756

import sys

def progressbar(it, prefix="Processing:", size=40, char='\u25b6', file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix + " ", char*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def test_progressbar():

    import time

    for i in progressbar(range(20), size=20):
        time.sleep(1)