from numba import njit, float32
import numpy as np


@njit(float32())
def get_betai():
    beta = np.ones(5, dtype="float32")
    return beta[1]


print(type(get_betai()))  # float64
print(type(get_betai.py_func()))  # float32


@njit()
def get_betai():
    beta = np.ones(5, dtype="float32")
    return beta[1:2]


print(get_betai().dtype)  # float32
print(get_betai.py_func().dtype)  # float32

###########################
# With int and float
###########################
print("########### Int and float")
X = np.arange(10).reshape(5, 2)
n, p = X.shape
arr = np.ones(2, dtype="float32")


@njit(float32())
def prod_intfloat():
    return n * arr[1]


print(type(prod_intfloat()))  # float64
print(type(prod_intfloat.py_func()))  # float64


@njit()
def prod_intfloat():
    return n * arr[1:2]


print(prod_intfloat().dtype)  # float32
print(prod_intfloat.py_func().dtype)  # float32
