import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def about_meshgrid():
    nx, ny = (3, 2)
    # x = np.linspace(0, 3, nx)
    x = np.array([1, 2, 2, 4, 5])
    y = np.linspace(0, 1, 15)
    xv, yv = np.meshgrid(x, y)
    print(x)
    print()
    print(xv)
    print()
    print(xv.shape)

    print()
    return

    print(y)
    print()
    print(yv)


def about_np_c_():

    a = np.array([[0, 1], [2, 3]])
    b = np.array([[4, 5], [6, 7]])

    print(a.flatten())
    # print(a)
    print()
    print(b.flatten())
    # print(b)
    print()

    c = np.c_[a.flatten(), b.flatten()]
    # c = np.c_[a, b]
    print(c)
    print()
    print(c.shape)
    print()


def about_torch_randn():
    # generate number from N(0, 1)
    y = torch.randn(1000)
    print(y.mean())
    print(y.std())


def about_itertool_count():
    from itertools import count

    for each in count(2):
        print(each)


def tensor_add():
    a = torch.tensor([1, 2])
    b = torch.tensor([3, 4])

    print(a.data)
    print()

    # print(a.add(b))
    # the following line will do an inplace add
    print(a.add_(b))
    print()

    print(a)
    print()
    print(b)
    print()


def about_smooth_l1_loss(verbose=False):

    POLY_DEGREE = 4
    torch.manual_seed(69294)

    W_target = torch.randn(POLY_DEGREE, 1) * 5
    b_target = torch.randn(1) * 5

    def make_features(x):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        x = x.unsqueeze(1)
        return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)

    def f(x):
        """Approximated function."""
        return x.mm(W_target) + b_target.item()

    def poly_desc(W, b):
        """Creates a string description of a polynomial."""
        result = 'y = '
        for i, w in enumerate(W):
            result += '{:+.2f} x^{} '.format(w, i + 1)
        result += '{:+.2f}'.format(b[0])
        return result

    def get_batch(batch_size=32):
        """Builds a batch i.e. (x, f(x)) pair."""
        random = torch.randn(batch_size)
        x = make_features(random)
        # generate labels
        y = f(x)
        return x, y

    train_x, train_y = get_batch(5)

    from itertools import count

    fc = torch.nn.Linear(W_target.size(0), 1)
    # print(fc)
    if verbose:
        print("The original poly:")
        print(poly_desc(fc.weight.view(-1), fc.bias.view(-1)))
        print()
    for train_idx in count(1):
        # rate = float(1 / (train_idx + 0))

        fc.zero_grad()

        batch_x, batch_y = get_batch()
        y_pred = fc(batch_x)
        loss = F.smooth_l1_loss(y_pred, batch_y)
        loss.backward()

        # learn
        for param in fc.parameters():
            param.data.add_(-0.09 * param.grad.data)
            # param.data.add_(-1 * rate * param.grad.data)

        if train_idx % 50 == 0 and verbose:
            # print(f"At {train_idx}th iteration, the loss = {loss.item()}")
            print(f"At {train_idx}th iteration, the loss = {loss}")

        if loss < 1e-4:
            if verbose:
                print()
                print(f"After {train_idx} iterations, the loss = {loss}")

                print()
                print("The model trained:")
                print(poly_desc(fc.weight.view(-1), fc.bias.view(-1)))
                print()
                print("The target:")
                print(poly_desc(W_target.view(-1), b_target.view(-1)))
            break

    return fc


def about_sgx0():

    normal = np.random.multivariate_normal
    # funny
    sgx0 = normal([0., 0.], [[1, 0.], [0., 1]], 2)

    print(sgx0)
    plt.plot(sgx0, 'g.')
    plt.show()


# about the initial parameters in fully connected layer
def about_params_in_fc():
    fc = torch.nn.Linear(4, 1)
    weight = fc.weight
    print(weight)
    print(weight.grad)
    print()


    target = torch.tensor([1.])
    res = fc(torch.randn(4))
    # print()
    # print(target.size())
    # print(res.size())
    print()
    loss = F.smooth_l1_loss(res, target)
    loss.backward()
    print(weight.grad)
    print()


def about_torch_multinomial():
    
    # indexes= torch.linspace(0,sgx.shape[0]-1,steps=sgx.shape[0])
    indexes= torch.linspace(0, 9, steps=10).reshape(5, -1)
    print(indexes)

    indexes = torch.squeeze(indexes)
    print(indexes)

    # random_idx = torch.multinomial(indexes, 3)
    # print(random_idx)


if __name__ == "__main__":

    # about_meshgrid()
    # about_np_c_()

    # about_torch_randn()
    # about_itertool_count()

    # tensor_add()

    # This is the trained layer
    # fc = about_smooth_l1_loss()
    # print(fc)
    # print(fc.weight)
    # print(fc.bias)

    # about_sgx0()
    # about_params_in_fc()

    # about_torch_multinomial()

    # x = torch.zeros(2, 4, 1)
    # print(x)
    # print()
    # print(x.size())
    # print()
    # y = torch.squeeze(x, 2)
    # print(y)
    # print(y.size())
    print()
