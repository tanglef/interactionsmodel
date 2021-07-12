from interactionsmodel.utils.numba_fcts import cpt_norm as cpt_norm_nb
from interactionsmodel.utils.numba_fcts import cpt_mean_std as cpt_mean_std_nb
from interactionsmodel.utils.numba_fcts import cpt_alpha_max as cpt_alpha_max_nb
from interactionsmodel.utils.numba_fcts import (
    dualgap_enet_inter as dualgap_enet_inter_nb,
)
from interactionsmodel.utils import preprocess, whitening
import torch
from interactionsmodel.utils import PRODS


def cpt_norm(
    X,
    X_cpt_Z,
    mean_Z,
    std_Z,
    n_features,
    n_squarefeatures,
    fit_interaction=True,
    inter_only=False,
    bind="numpy",
    flatten=False,
    full=False,
    zca=False,
):
    """Compute the squared norm of each feature for the data X and the
    interaction matrix Z/

    Args:
        X (array): Standardized data
        X_cpt_Z (arary): Original data
        mean_Z (array): Means of the features in Z
        std_Z (array): Standard deviation of the features in Z
        n_features (int): Number of features for X
        n_squarefeatures (int): Number of features for Z
        fit_interaction (bool, optional): Comput the norm for Z.
             Defaults to True.
        inter_only (bool, optional): Include squared termis in interactions.
             Defaults to False.
        bind (str): "torch" if working with torch, else numpy is used.
        flatten (bool, optional): when using torch, return a list by block
            for Znorm2 or a usual list. Defaults to False (by block).
    """
    if bind != "torch":
        return cpt_norm_nb(
            X,
            X_cpt_Z,
            mean_Z,
            std_Z,
            n_features,
            n_squarefeatures,
            fit_interaction,
            inter_only,
            full,
            zca,
        )
    else:
        p = X.shape[1]
        X_cpt_Z = X if zca else X_cpt_Z
        X_norm2 = torch.linalg.norm(X, 2, axis=0) ** 2
        Z_norm2 = []
        if fit_interaction:
            for var in range(p):
                Xi = X_cpt_Z[:, var].view(-1, 1)
                K = ((Xi * X_cpt_Z[:, var:]) - mean_Z[var]) / std_Z[var]
                if flatten:
                    Z_norm2.extend(torch.linalg.norm(K, 2, axis=0) ** 2)
                else:
                    Z_norm2.append(torch.linalg.norm(K, 2, axis=0) ** 2)
        return X_norm2, Z_norm2


def cpt_mean_std(
    X,
    inter_only=False,
    fit_interaction=True,
    standardize=True,
    inter_thn_std=True,
    full=False,
    bind="numpy",
):
    """Compute the mean and std of each column of X and/or Z

    Note that arguments inter_only, fit_interaction, standardize,
    inter_thn_std are only for numpy binding. The argument full is
    only for the torch binding.
    When using torch binding, mean_Z and std_Z are lists of lists.
    Each sublist corresponds to the values for a block of Z.

    Args:
        X (array): data
        inter_only (bool, optional): use squared terms. Defaults to False.
        fit_interaction (bool, optional): compute for Z?. Defaults to True.
        standardize (bool, optional): do the computation. Defaults to True.
        inter_thn_std (bool, optional): Defaults to True.
        full (bool, optional): Take p**2 interactions. Defaults to False.
        bind (str, optional): Use numpy of torch. Defaults to "numpy".

    Returns:
        mean_X, std_X: means and std of X
        mean_Z, std_Z: means and std of Z

    """
    if bind != "torch":
        return cpt_mean_std_nb(
            X, inter_only, fit_interaction, standardize, inter_thn_std, full
        )
    else:
        return preprocess(X, full)


def dual_gap_enet(
    beta,
    theta,
    residuals,
    alpha_beta1,
    alpha_theta1,
    alpha_beta2,
    alpha_theta2,
    alpha,
    y,
    X,
    X_cpt_Z,
    mean_Z,
    std_Z,
    n_samples,
    n_features,
    dual,
    y_norm2,
    inter_only=False,
    bind="numpy",
    full=False,
):
    """Compute the dual gap.

    The inter_only option is only for the numpy binding.

    Args:
        binding (str, optional): Use torch or numpy. Defaults to "numpy".

    Returns:
        primal, dual, gap, nu, cst_dual
    """
    if bind != "torch":
        return dualgap_enet_inter_nb(
            beta,
            theta,
            residuals,
            inter_only,
            alpha_beta1,
            alpha_theta1,
            alpha_beta2,
            alpha_theta2,
            alpha,
            y,
            X,
            X_cpt_Z,
            mean_Z,
            std_Z,
            n_samples,
            n_features,
            dual,
            y_norm2,
            full,
        )
    else:
        dtype = X.dtype
        device = X.device
        primal = (
            0.5 * 1 / n_samples * torch.linalg.norm(residuals) ** 2
            + alpha_beta1 * torch.linalg.norm(beta, 1)
            + alpha_theta1 * torch.linalg.norm(theta, 1)
            + alpha_beta2 / 2 * torch.linalg.norm(beta) ** 2
            + alpha_theta2 / 2 * torch.linalg.norm(theta) ** 2
        )
        gap = torch.tensor([primal], device=device, dtype=dtype)

        # ||X||_infty
        norm_X = (
            alpha
            / alpha_beta1
            * torch.abs(X.T @ residuals - n_samples * alpha_beta2 * beta).max()
        )
        norm_Z = torch.tensor([-1e8], device=device, dtype=dtype)
        obegin, p = 0, X.shape[1]
        for var in range(n_features):
            if not full:
                p_tilde = p - var
                Zi = X_cpt_Z[:, var].view(-1, 1) * X_cpt_Z[:, var:]
            else:
                p_tilde = p
                Zi = X_cpt_Z[:, var].view(-1, 1) * X_cpt_Z
            Zi -= mean_Z[var]
            Zi /= std_Z[var]
            norm_Z = torch.max(
                norm_Z,
                torch.abs(
                    Zi.T @ residuals
                    - n_samples * alpha_theta2 * theta[obegin : (obegin + p_tilde)]
                ).max(),
            )
            obegin += p_tilde
        norm_Z = alpha / alpha_theta1 * norm_Z
        cst_dual = max(alpha * n_samples, norm_X, norm_Z)
        nu = residuals / cst_dual

        new_dual = (
            y_norm2 * 0.5 / n_samples
            - 0.5
            * n_samples
            * alpha ** 2
            * torch.linalg.norm(nu - y / (n_samples * alpha)) ** 2
            - (n_samples * alpha / cst_dual) ** 2
            * alpha_beta2
            / 2
            * torch.linalg.norm(beta) ** 2
            - ((n_samples * alpha) / cst_dual) ** 2
            * alpha_theta2
            / 2
            * torch.linalg.norm(theta) ** 2
        )
        if new_dual > dual:
            dual = new_dual
        gap -= dual
    return primal, dual, gap, nu, cst_dual


def cpt_lambda_max(
    X,
    X_cpt_Z,
    y,
    mean_X,
    mean_Z,
    std_X,
    std_Z,
    l1_ratio=1,
    inter_only=False,
    fit_intercept=True,
    fit_interaction=True,
    standardize=True,
    n_breaks=20,
    bind="numpy",
    zca=False,
):
    if bind != "torch":
        return cpt_alpha_max_nb(
            X,
            y,
            mean_X,
            mean_Z,
            std_X,
            std_Z,
            l1_ratio,
            inter_only,
            fit_intercept,
            fit_interaction,
            standardize,
            zca=zca,
        )
    else:
        _, prodT = PRODS["small"]
        n, _ = X.shape
        lambda1_max = torch.max(torch.abs(X.T @ y)) / (n * l1_ratio)
        if fit_interaction:
            res = prodT(X_cpt_Z, y, mean_Z, std_Z, n_breaks)
            lambda2_max = torch.max(torch.abs(res)) / (n * l1_ratio)
        else:
            lambda2_max = torch.tensor([0]).type(X.type())
    return lambda1_max, lambda2_max
