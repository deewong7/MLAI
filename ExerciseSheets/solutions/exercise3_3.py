from exercise2_1 import get_entropy
import data
from typing import Tuple

entries = []

exercise = ["EXERCISE", "daily", "weekly", "rarely"]
family = ["FAMILY", "yes", "no"]
smoker = ["SMOKER", "true", "false"]
obese = ["OBESE", "true", "false"]

risk = ["RISK", "low", "high"]


AMOUNT = 5


def display_lst(lst):
    for each in lst:
        print(each)
    print()


def read_raw(sample: str, disp: bool = True, calculate_e: bool = True) -> Tuple[list, float]:
    rows = []
    e = 0.0
    for each in sample.splitlines():
        row = []

        if each == "":
            continue

        for item in each.split():
            if item.isdigit():
                item = int(item)

            row.append(item)

        rows.append(row)

    if disp:
        display_lst(rows)

    if calculate_e:
        e = print_overall_entropy(rows)

    return rows, e


def _get_entropy(c1: dict) -> float:
    return get_entropy(c1, AMOUNT)


def print_overall_entropy(rows) -> float:
    overall_count = {risk[1]: 0, risk[2]: 0}
    for each in rows:
        overall_count[each[3]] += 1

    overall_entropy = _get_entropy(overall_count)
    print("Overall Entropy = %.4f" % overall_entropy)
    print()

    return overall_entropy


if __name__ == "__main__":

    def split_by(feature, rows, idx) -> float:

        print("Split by", feature[0] + ":")
        print()

        entropy = 0.0
        for i in range(1, len(feature)):
            current_f = feature[i]

            count = {}
            # find the amount according to each specific feature
            for each in rows:
                value = each[idx]
                output = each[-1]

                if value in feature and value == current_f:
                    if output in count:
                        count[output] += 1
                    else:
                        count[output] = 1

            print("Value =", current_f, ":", count)
            e = _get_entropy(count)
            entropy += e
            print("Weighted entropy = %.4f" % e, end="\n\n")

        print("Overall Entropy for (%s) = %.4f" %
              (feature[0], entropy), end="\n\n")

        return float("%.4f" % entropy)

    # Bootstrap Sample A
    def bootstrap_sample_a():
        print("Bootstrap Sample A", end="\n\n")
        rows, e = read_raw(data.raw_bootstrap_sample1)
        e1 = split_by(exercise, rows, 1)
        e2 = split_by(family, rows, 2)
        print("Conclusion:")
        print("Information Gain (%s)= %.4f" % (exercise[0], e - e1))
        print("Information Gain (%s)= %.4f" % (family[0], e - e2), end="\n\n")

    # Bootstrap Sample B
    def bootstrap_sample_b():
        print("Bootstrap Sample B", end="\n\n")
        rows, e = read_raw(data.raw_bootstrap_sample2)
        e1 = split_by(smoker, rows, 1)
        e2 = split_by(obese, rows, 2)
        print("Conclusion:")
        print("Information Gain (%s)= %.4f" % (smoker[0], e - e1))
        print("Information Gain (%s)= %.4f" % (obese[0], e - e2), end="\n\n")

    # Bootstrap Sample C
    def bootstrap_sample_c():
        print("Bootstrap Sample C", end="\n\n")
        rows, e = read_raw(data.raw_bootstrap_sample3)
        e1 = split_by(obese, rows, 1)
        e2 = split_by(family, rows, 2)
        print("Conclusion:")
        print("Information Gain (%s)= %.4f" % (obese[0], e - e1))
        print("Information Gain (%s)= %.4f" % (family[0], e - e2), end="\n\n")

    # bootstrap_sample_a()
    # bootstrap_sample_b()
    bootstrap_sample_c()
