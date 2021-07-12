"""
==============================================================
Standardization utilities for the interaction matrix by block
==============================================================
"""

import torch


def preprocess(X, full=False):
    std_X, mean_X = torch.std_mean(X, 0, unbiased=False)
    p = X.shape[1]
    means_bb = []
    std_bb = []
    for var in range(p):
        if full:
            Z_bb = X[:, var].view(-1, 1) * X
        else:
            Z_bb = X[:, var].view(-1, 1) * X[:, var:]
        means_bb.append(torch.mean(Z_bb, 0))
        std_bb.append(torch.std(Z_bb, 0, unbiased=False))
    return mean_X, std_X, means_bb, std_bb
