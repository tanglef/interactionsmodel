import torch
import numpy as np


def make_Z_full(X, bind="torch"):
    n, p = X.shape
    if bind == "torch":
        for var in range(p):
            Xi = X[:, var].view(-1, 1)
            if var == 0:
                Z = Xi * X
            else:
                Z = torch.cat((Z, Xi * X), dim=1)
    else:
        Z = np.zeros(shape=(n, int(p ** 2)))
        jj = 0
        for j1 in range(p):
            for j2 in range(p):
                Z[:, jj] += X[:, j1] * X[:, j2]
                jj += 1
    return Z


def make_Z(X, bind="torch"):
    n, p = X.shape
    if bind == "torch":
        for var in range(p):
            Xi = X[:, var].view(-1, 1)
            if var == 0:
                Z = Xi * X[:, var:]
            else:
                Z = torch.cat((Z, Xi * X[:, var:]), dim=1)
    else:
        Z = np.zeros(shape=(n, int(p * (p + 1) / 2)))
        jj = 0
        for j1 in range(p):
            for j2 in range(j1, p):
                Z[:, jj] += X[:, j1] * X[:, j2]
                jj += 1
    return Z
