import torch

a = torch.rand((3, 5))
b = torch.rand((3, 2))
c = torch.rand((2))


def about_matrix_multiplication():
    # matrix multiplication

    res = []
    res.append(a @ b)
    res.append(torch.matmul(a, b))
    res.append(torch.mm(a, b))

    for each in res:
        for i in range(len(res)):
            if res[i] is not each:
                print(torch.eq(each, res[i]))


def about_torch_dimension():
    print(a.dim())
    print(b.dim())

    print(c.dim())
    print(c)


def about_concatenate():
    print()
    print(a)
    print()
    print(b)
    print()

    d = torch.cat([a, b], dim = 1) # will cat through rows
    print(d.shape)
    print(d)
    print()


def about_torch_randn():
    # the larger, the more similar to N(0, 1)
    # where mean = 0 and variance = 1
    w = torch.randn([50000, 3])
    w /= 10
    print(w)
    print(w.var())
    print(w.std())

if __name__ == "__main__":
    # about_torch_dimension()
    # about_concatenate()
    about_torch_randn()
