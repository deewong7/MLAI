import numpy

arr = [[1., 0., 0.], [0., 1., 2.]]
int_arr = [[1, 0, 0], [0, 1, 2]]
int_arr = [[1, 0, 0], [0, 1, 2]]

narr = numpy.array(arr)
narr = numpy.array(int_arr, dtype="int32")

# print(type(narr))
# print(narr.shape)
# print(narr.max())
# print(narr.min())
# print(narr.ndim)
# print(narr.dtype)
# print(narr.itemsize)
# print(narr.data)

def analyse(sequence):
    print()
    print(sequence)
    print(type(sequence))
    print()

zeros = numpy.zeros((5, 5), dtype=numpy.int32)
# analyse(zeros)

sequences = numpy.arange(3, dtype="float")
sequences = numpy.arange(3, 10, 5)
# analyse(sequences)

float_sequences = numpy.linspace(1, 10, 5)
# analyse(float_sequences)

a = [[1, 2], [3, 4]]
# analyse(a)
zeros_like = numpy.zeros_like(a)
ones_like = numpy.ones_like(a)
# analyse(zeros_like)
# analyse(ones_like)

# reshape

casual = numpy.arange(10)
casual = numpy.zeros(10, numpy.int8).reshape(5, 2)
# analyse(casual)

# print( numpy.int0 == numpy.int64 ) # True

# print((casual + 2) ** 4)
A = numpy.array([[1, 2], [3, 4]])
B = numpy.array([[5, 6], [7, 8]])
C = A @ B

# analyse(A)
# analyse(B)
# analyse(C)

# axis = 0 -> column
# axis = 1 -> row

# a = numpy.random.random((2, 3))
a = numpy.array(a)
# analyse(a)
# print(a.sum(axis=0), end="\n\n") # row
# print(a.sum(axis=1), end="\n\n") # column

def f(x, y):
    return 10 * x + y

def f2(x, y, z):
    return 100 * x + 10 * y + z

from_f = numpy.fromfunction(f, (5, 5), dtype=int) # calculation
# from_f = numpy.fromfunction(f2, (5, 5, 5), dtype=int) # calculation
# analyse(from_f)

# print(from_f[-1]) # equal to from_f[-1, :] -> the last row of all columns
# print(from_f[:, -1]) # -> the last column of all rows
# print(from_f[..., -1]) # -> the last column of all rows

# for i in range(5):
#     for j in range(5):
#         print(i, "* 10 +", j, "=", from_f[i, j])

# for item in from_f.flat:
    # print(item)

# print(from_f.T)

a = numpy.floor(10 * numpy.random.random((2, 2)))
b = numpy.floor(10 * numpy.random.random((2, 2)))
c = numpy.vstack((a, b))
c = numpy.hstack((b, a))
c = c.astype(numpy.int0)

# analyse(a)
# analyse(b)
analyse(c)
e = c.view()
d = c.copy()
# analyse(d)
# del c
# alyse(d)

# print(d == e)
# print(e.base is c)
# print(d.base is c)