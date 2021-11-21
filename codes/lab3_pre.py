from util import analyse
from sklearn.model_selection import KFold, PredefinedSplit
import numpy as np
from typing import Generator


def about_predefined_split():
    from sklearn.model_selection import PredefinedSplit
    X = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
    y = np.array([0, 0, 1, 1])
    test_fold = [-1, 1, 4, 2]
    ps = PredefinedSplit(test_fold)
    ps.get_n_splits()

    # print(ps)

    i = 1
    for train_index, test_index in ps.split():
        print(i)
        print("TRAIN:\n", X[train_index], "\nTEST:\n", X[test_index])
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        i += 1


def about_stack_ndarray():
    a = np.array([1, 2, 3])
    print("a.shape:", a.shape)
    b = np.array([2, 3, 4, 5])
    print("b.shape:", b.shape)

    c = np.hstack([a, b])
    print("c", c)
    print("c.shape", c.shape)


def about_diff_between_kfold_and_predefinedsplit():
    # the goal for theses is the same, to split the data set into 2 sets, train set and test set
    # to explore the difference or/and behavior between the KFold and PredefinedSplit when get_n_split() returns 2

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])
    print("X:\n", X)
    print("y\n", y)

    def result(name: str, generator: Generator):
        print(">>>\n" + name + ":\n")
        i = 1
        for train_indexes, test_indexes in generator:
            print(">>>\nSplit", i, ":\n")
            # print("Train indexes:", train_indexes)
            print("X for train:", X[train_indexes])
            print("y for train:", y[train_indexes])
            print()
            # print("Test indexes", test_indexes)
            print("X for test:", X[test_indexes])
            print("y for test:", y[test_indexes])
            print("\nEnd of Split %d." % i)
            print("<<<\n")
            i += 1

    # about KFold

    def about_KFold():
        kf = KFold(n_splits=2)
        # print(kf.get_n_splits(X)) # there will be 2 splits available

        split_data_generator = kf.split(X=X, y=y)
        # the generator has finished its task
        split_data_list = list(split_data_generator)

        def explore_with_out_a_for_loop():

            split1 = split_data_list[0]
            train_set_indexes_for_split1 = split1[0]
            test_set_indexes_for_split1 = split1[1]

            X_train1 = X[train_set_indexes_for_split1]
            y_train1 = y[train_set_indexes_for_split1]
            print("X_train1:", X_train1)
            print("y_train1:", y_train1)

            X_test1 = X[test_set_indexes_for_split1]
            y_test1 = y[test_set_indexes_for_split1]
            print("X_test1:", X_test1)
            print("y_test1:", y_test1)

        result("KFold", kf.split(X, y))

    # about PredefinedSplit

    def about_predefined_split():

        # test_fold[i] = index of the test set that sample i belongs to
        # In another word, in each of the fold there need to be a test set
        # which sample should be the test set? Their indexes has already defined
        # in the list called test_fold

        test_fold = [-1, -1, -1, 0]
        # sample 0, 1, 2 never go into test set.

        test_fold = [3, 2, 1, 0]
        # test_fold[0] = 3, sample 0 is the test set in split 4
        # test_fold[1] = 2, smaple 1 is the test set in split 3
        # test_fold[2] = 1, smaple 2 is the test set in split 2
        # test_fold[3] = 0, smaple 2 is the test set in split 1

        test_fold = [1, 1, 1, 1]
        # there will be only one split since sample i = 1, in split 1, the test indexes is [0, 1, 2, 3]
        test_fold = [3, 2, 1, 1]
        # there will be 3 split:
        # In split 1, test indexes is [2, 3]
        # In split 2, test indexes is [1]
        # In split 3, test indexes is [0]

        test_fold = [0, 1, -1, 1]
        # there will 2 splits
        # In the first split (split0), the test indexes is [0]
        # In the second split (split1), the test indexes is [1, 3]
        # -1: the sample 2 will never go into test set

        # ~tricky
        test_fold = [0, 2, -1, 2]  # the same as [0, 1, -1, 1]

        # Bingo: this test_fold will work the same as KFold(n=2)
        test_fold = [0, 0, 1, 1]

        ps = PredefinedSplit(test_fold)
        # print("get_n_splits() = ", ps.get_n_splits())
        result("PredefinedSplit", ps.split())

        # Is it necessary to give y as parameter?
        # No, parameter X and y are ignored in PredefinedSplit.split()
        # but they are not ignored in KFold.split(X, y)

    about_KFold()
    about_predefined_split()


if __name__ == "__main__":
    pass
    about_diff_between_kfold_and_predefinedsplit()
