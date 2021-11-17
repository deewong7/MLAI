import numpy as np


def q2():

    x1 = 3
    x2 = 5
    x3 = 1

    e = 1e-6

    def f1(x1, x2, x3) -> float:
        return x1 * x3 + np.log(x2 + x1) * np.exp(-x3)

    def f2(x1, x2, x3) -> float:
        return np.exp(-x2) + np.cos(x1 * x3)

    j11 = (f1(x1, x2, x3) - f1(x1 - e, x2, x3)) / e
    j12 = (f1(x1, x2, x3) - f1(x1, x2 - e, x3)) / e
    j13 = (f1(x1, x2, x3) - f1(x1, x2, x3 - e)) / e

    j21 = (f2(x1, x2, x3) - f2(x1 - e, x2, x3)) / e
    j22 = (f2(x1, x2, x3) - f2(x1, x2 - e, x3)) / e
    j23 = (f2(x1, x2, x3) - f2(x1, x2, x3 - e)) / e

    print(j11)
    print(j12)
    print(j13)

    print(j21)
    print(j22)
    print(j23)

if __name__ == "__main__":
    q2()
