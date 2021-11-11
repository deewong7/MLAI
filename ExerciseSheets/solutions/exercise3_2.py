import math, statistics
from exercise2_1 import init_rows, combine_dict, get_entropy

A = "PASS"
B = "FAIL"

raw_rows = """
1 yes tired 65
2 no alert 20
3 yes alert 90
4 yes tired 70
5 no tired 40
6 yes alert 85
7 no tired 35
"""
AMOUNT = 7

def display_lst(lst):
    for each in lst:
        print(each)
    print()


if __name__ == "__main__":

    def split_by(feature, index, postive_value):
        print("Split by", feature + ":")
        print()

        positives = []
        negatives = []

        for each in rows:
            score = int(each[3])
            if each[index] == postive_value:
                positives.append(score)
            else:
                negatives.append(score)

        var_pos = statistics.variance(positives)
        var_neg = statistics.variance(negatives)

        print("Positive variance = % 10.4f, weighting = %d/%d" % (var_pos, len(positives), AMOUNT))
        print("Negative variance = % 10.4f, weighting = %d/%d" % (var_neg, len(negatives), AMOUNT))
        print("\nWeighted Variance = % 10.4f" % ((var_pos * len(positives) / AMOUNT) + (var_neg * len(negatives) / AMOUNT)))
        print()


    rows = init_rows(raw_rows, sort_by_index=3)
    display_lst(rows)

    split_by("STUDIED", 1, "yes")
    split_by("ENERGY", 2, "alert")

