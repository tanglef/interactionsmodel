from interactionsmodel.utils import PRODS
import torch


def power_method(X, which, meanZ=None, stdZ=None, full=False, maxiter=100, eps=1e-5):
    """Computes spectral ray of X.T @ X or Z.T @ Z

    Args:
        X (tensor): data of size (n,p)
        which (str):
            - "X" to compute the spectral ray of X.T @ X
            - "Z" to compute the spectral ray of Z.T @ Z
        full (bool, optional):
            - False: use unique interactions in Z,
                resulting in using a matrix of size (n,p*(p+1)/2),
            - True: use all interactions in Z,
                resulting in using a matrix of size (n,p**2).
            Defaults to False.
        maxiter (int, optional): Number of iterations for the power method.
            Defaults to 500.
    """
    p = X.shape[1]
    if which == "X":
        XT = X.T

        def prod(X, x, *args):
            return X @ x

        def prodT(X, xt, *args):
            return XT @ xt

        size = p
    else:
        prods = PRODS["full"] if full else PRODS["small"]
        prod = prods[0]
        prodT = prods[1]
        size = int(p * (p + 1) / 2) if not full else int(p ** 2)

    device = X.device
    z = torch.randn((size, 1), dtype=X.dtype, device=device)

    z /= torch.linalg.norm(z)
    val = 0
    for k in range(maxiter):
        z_new = prodT(X, prod(X, z, meanZ, stdZ), meanZ, stdZ)
        # Rayleigh quotient
        val_new = val
        val = z_new.view(-1) @ z.view(-1)
        if ((val - val_new) ** 2).sum() <= eps ** 2:
            break
        z = z_new / torch.linalg.norm(z_new)
    if k + 1 == maxiter:
        print("Warning ----------- Power iteration method did not converge !")
    return val


def soft_thresh(x, lambda_):  # should I use torch.clamp ?
    """Soft thresholding operator

    Args:
        x (array): vector on which to apply the thresholding
        lambda_ (float): threshold

    Returns:
        (array): ST(x, lambda)=sign(x) (|x| - lambda)_+
    """
    return torch.sign(x) * torch.maximum(torch.abs(x) - lambda_, torch.zeros_like(x))


def Lanczos(X, which, meanZ=None, stdZ=None, n_cv=20, full=False):
    n, p = X.shape
    if which == "X":
        XT = X.T

        def prod(X, x, *args):
            return X @ x

        def prodT(X, xt, *args):
            return XT @ xt

        size = p
    elif which == "Z":
        prods = PRODS["full"] if full else PRODS["small"]
        prod = prods[0]
        prodT = prods[1]
        size = int(p * (p + 1) / 2) if not full else int(p ** 2)

    m = min(size, n_cv) if size > 1 else size
    T = torch.zeros((m, m), dtype=X.dtype, device=X.device)
    v0 = torch.randn((size, 1), dtype=X.dtype, device=X.device)
    v0 /= torch.linalg.norm(v0)
    w = prodT(X, prod(X, v0, meanZ, stdZ), meanZ, stdZ)
    alpha = w.flatten() @ v0.flatten()
    w -= alpha * v0
    T[0, 0] = alpha
    for j in range(1, m):
        beta = torch.linalg.norm(w)
        T[j - 1, j] = T[j, j - 1] = beta
        v1 = w / beta
        w = prodT(X, prod(X, v1, meanZ, stdZ), meanZ, stdZ)
        alpha = w.view(-1) @ v1.view(-1)
        w -= alpha * v1 + beta * v0
        v0 = v1.clone()
        T[j, j] = alpha
    return torch.linalg.norm(T, 2)
