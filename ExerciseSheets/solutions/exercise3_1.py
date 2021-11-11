# threshold
from math import log2

threshold = 45  # 26, 39.5, 40.5

entries = {}  # id -> rows, to be generated

A = "≥50K"
B = "25K-50K"
C = "≤25K"
raw_rows = """
1 39 bachelors never married transport 25K-50K
2 50 bachelors married professional 25K-50K
3 18 high school never married agriculture ≤25K
4 28 bachelors married professional 25K-50K
5 37 high school married agriculture 25K-50K
6 24 high school never married armed forces ≤25K
7 52 high school divorced transport 25K-50K
8 40 doctorate married professional ≥50K
"""
AMOUNT = 8

# raw_columns = "ID AGE EDUCATION MARITAL STATUS OCCUPATION ANNUAL INCOME"
# columns = []

# for each in raw_columns.split():
#     columns.append(each)
# print(columns)


def _generate_entries(row: str):

    entry = []
    for each in row.split():
        if each.isdigit():
            each = int(each)

        entry.append(each)

    # print(entry)
    entries[entry[0]] = entry
    # print(entries)


def _generate(raw_rows: str):

    for row in raw_rows.splitlines():
        if row == "":
            continue

        _generate_entries(row)


def init_rows(raw_rows, sort_by_index: int = 0, need_purify: bool = False, one: int = 1, other: int = -1) -> list:

    _generate(raw_rows)
    rows = list(entries.values())
    rows.sort(key=lambda x: x[sort_by_index])

    # for exercise 1c
    def purify(lst, one, other) -> list:
        if need_purify:
            new_lst = []
            new_lst.append(lst[one])
            new_lst.append(lst[other])

        return new_lst

    if need_purify:
        i = 0
        for each in rows:
            rows[i] = purify(each, one, other)  # in-place rewrite
            i += 1

    return rows


def _find_count(lst) -> dict:
    count = {A: 0, B: 0, C: 0}
    count[lst[1]] += 1

    return count


def combine_dict(old: dict, new: dict):
    for k in new.keys():
        if k in old:
            value = old[k] + new[k]
        else:
            value = new[k]
        old[k] = value


def entropy(p: float) -> float:
    if p != 0:
        return ((-1) * p * log2(p))
    else:
        return 0


def get_entropy(count: dict, AMOUNT: int, decimal=4) -> float:
    p = 0.0
    total = float(sum(count.values()))

    for each in count.values():
        p += entropy(float(each / total))

    if total != AMOUNT:
        weighting = total / AMOUNT
        print("(Partition Entropy = %.4f, weighting = %d/%d)" % (p, total, AMOUNT))
        p *= weighting

    p = float(("{:." + str(decimal) + "f}").format(p))
    # p = float(("%.4f" % p))
    return p


def _welcome():
    print("There are", AMOUNT, "rows.")
    print("Threshold is set to", str(threshold) + ".")
    print()


def _get_entropy(c1) -> float:
    return get_entropy(c1, AMOUNT)


def _report(c1: dict, c2: dict, ENTROPY: float):
    print()
    print("After splitting:")
    print()

    print(c1, end=":\nRemainder = ")
    e1 = _get_entropy(c1)
    print("%.4f" % e1)

    print()

    print(c2, end=":\nRemainder = ")
    e2 = _get_entropy(c2)
    print("%.4f" % e2)

    print()
    print("Remainder = %.4f" % (e1 + e2))
    print("Information Gain = %.4f" % (ENTROPY - e1 - e2), end="\n\n")


def display_with_threshold(rows):
    split = False
    for each in rows:
        if each[0] >= threshold and not split:
            print("--------------")
            split = True
        print(each)
    print()


if __name__ == "__main__":

    count = {}
    count2 = {}
    rows = init_rows(raw_rows, sort_by_index=1, need_purify=True)
    _welcome()
    display_with_threshold(rows)

    # Overall Entropy:
    overall_count = {}
    for each in rows:
        combine_dict(overall_count, _find_count(each))
    ENTROPY = get_entropy(overall_count, AMOUNT)

    # split by threshold
    for each in rows:
        if each[0] <= threshold:
            combine_dict(count, _find_count(each))
        else:
            combine_dict(count2, _find_count(each))

    _report(count, count2, ENTROPY)
