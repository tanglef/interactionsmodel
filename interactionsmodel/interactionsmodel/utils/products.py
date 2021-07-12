"""Regroup the different matvec products"""
import torch


def product_Z(X, beta, means, std):
    """Make the product Z @ beta by blocks w/ torch."""
    n, p = X.shape
    obegin = 0
    res = torch.zeros((n, 1), dtype=beta.dtype, device=beta.device)
    for var in range(p):
        p_tilde = p - var
        Xi = X[:, var].view(-1, 1)
        K = ((Xi * X[:, var:]) - means[var]) / std[var]
        res += (
            K
            @ beta[
                obegin : (obegin + p_tilde),
            ].view(-1, 1)
        )
        obegin += p_tilde
    return res


def product_ZT(X, theta, means, std, n_breaks=1):
    """Make the product Z.T @ beta by blocks w/ torch."""
    n, p = X.shape
    q = int(p * (p + 1) / 2)
    breaks = torch.linspace(0, n, n_breaks + 1, dtype=torch.int)
    obegin = 0
    theta = theta.view(1, -1)
    res = torch.zeros((q, 1), dtype=theta.dtype, device=theta.device)
    for var in range(X.shape[1]):
        p_tilde = p - var
        for k, nk in enumerate(breaks[:-1]):
            nk1 = breaks[k + 1]
            xi = X[nk:nk1, var].view(-1, 1)
            Z = xi * X[nk:nk1, var:]
            Z -= means[var]
            Z /= std[var]
            res[obegin : (obegin + p_tilde)] += Z.T @ theta[:, nk:nk1].view(-1, 1)
        obegin += p_tilde
    return res


def product_Z_full(X, beta, means, std):
    """Make the product Z @ beta w/ doubles by blocks w/ torch."""
    n, p = X.shape
    obegin = 0
    res = torch.zeros((n, 1), dtype=beta.dtype, device=beta.device)
    for var in range(p):
        Xi = X[:, var].view(-1, 1)
        K = ((Xi * X) - means[var]) / std[var]
        res += (
            K
            @ beta[
                obegin : (obegin + p),
            ].view(-1, 1)
        )
        obegin += p
    return res


def product_ZT_full(X, theta, means, std, n_breaks=1):
    """Make the product Z.T @ theta w/ doubles by blocks w/ torch."""
    n, p = X.shape
    breaks = torch.linspace(0, n, n_breaks + 1, dtype=torch.int)
    q = int(p ** 2)
    obegin = 0
    theta = theta.view(1, -1)
    res = torch.zeros((q, 1), dtype=theta.dtype, device=theta.device)
    for var in range(X.shape[1]):
        for k, nk in enumerate(breaks[:-1]):
            nk1 = breaks[k + 1]
            xi = X[nk:nk1, var].view(-1, 1)
            Z = xi * X[nk:nk1, :]
            Z -= means[var]
            Z /= std[var]
            res[obegin : (obegin + p)] += Z.T @ theta[:, nk:nk1].view(-1, 1)
        obegin += p
    return res
