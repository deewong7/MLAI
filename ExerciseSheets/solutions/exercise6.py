import numpy as np

def postive_p(w: np.ndarray, x: np.ndarray):
    return ( 1 / (1 + np.exp(- w.T @ x)) )


def q4():
    w = np.array([4, -2, 5, -3, 11, 9])
    x = np.array([6, 8, 2, 7, -3, 5])

    print(- w.T @ x)

    print(postive_p(w, x))



if __name__ == "__main__":
    q4()
