import torch
import numpy as np


def whitening(X, eps=1e-1, bind="numpy"):
    sigma = X.T @ X / X.shape[0]
    if bind != "torch":
        U, S, _ = np.linalg.svd(sigma, full_matrices=False)
        mat_zca = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    else:
        U, S, _ = torch.svd(sigma, some=True)
        mat_zca = U @ torch.diag(1.0 / torch.sqrt(S + eps)) @ U.T
    return mat_zca
