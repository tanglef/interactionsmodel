from numpy.linalg import norm
from numba import njit
import numpy as np


@njit(fastmath=True)
def soft_thresholding(x, alpha, dtype="float64"):
    """
    This function does a soft thresholding of x in function of alpha.

    Input :
    x : a 1-D numpy array (numeric)
    alpha : a scalar (numeric)

    Output:
    result : a 1-D numpy array (numeric), result of
             the soft thresholding operation
    """
    if x > alpha:
        return x - alpha
    if x < -alpha:
        return x + alpha
    return 0


@njit()
def cd_enet_inter(
    beta,
    theta,
    residuals,
    inter_only,
    alpha1,
    alpha2,
    X,
    X_cpt_Z,
    X_norm2,
    X_norm2_alpha_inv,
    mean_Z,
    std_Z,
    Z_norm2,
    Z_norm2_alpha_inv,
    n_samples,
    n_features,
    dtype="float64",
    full=False,
):
    jj = 0

    beta_theta_max = 0
    d_beta_theta_max = 0

    for j1 in range(n_features):
        new_beta = (
            soft_thresholding(
                residuals.T @ X[:, j1] + beta[j1] * X_norm2[j1],
                (n_samples * alpha1),
                dtype=dtype,
            )
            * X_norm2_alpha_inv[j1]
        )
        diff_beta = beta[j1] - new_beta

        d_beta_theta_max = max(d_beta_theta_max, abs(diff_beta))
        beta_theta_max = max(beta_theta_max, abs(new_beta))

        if diff_beta != 0:
            residuals += diff_beta * X[:, j1]
            beta[j1] = new_beta

        j_int = j1 + inter_only if not full else 0
        for j2 in range(j_int, n_features):
            Z_tmp = (X_cpt_Z[:, j2] * X_cpt_Z[:, j1] - mean_Z[jj]) / std_Z[jj]
            new_theta = (
                soft_thresholding(
                    residuals.T @ Z_tmp + theta[jj] * Z_norm2[jj],
                    (n_samples * alpha2),
                    dtype=dtype,
                )
                * Z_norm2_alpha_inv[jj]
            )
            diff_theta = theta[jj] - new_theta

            d_beta_theta_max = max(d_beta_theta_max, abs(diff_theta))
            beta_theta_max = max(beta_theta_max, abs(new_theta))

            if diff_theta != 0:
                residuals += diff_theta * Z_tmp
                theta[jj] = new_theta
            jj += 1
    return (beta, theta, residuals, beta_theta_max, d_beta_theta_max)


@njit()
def dualgap_enet_inter(
    beta,
    theta,
    residuals,
    inter_only,
    alpha1,
    alpha2,
    alpha3,
    alpha4,
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
    full=False,
):
    primal = (
        (1 / (2 * n_samples)) * norm(residuals) ** 2
        + alpha1 * norm(beta, 1)
        + alpha2 * norm(theta, 1)
        + (alpha3 / 2) * norm(beta) ** 2
        + (alpha4 / 2) * norm(theta) ** 2
    )
    gap = primal

    # compute the X infinity norm
    norm_X = (alpha / alpha1) * np.abs(
        X.T @ residuals - n_samples * alpha3 * beta
    ).max()

    # compute the Z infinity norm
    norm_Z = -np.infty
    jj = 0
    for j1 in range(n_features):
        j_int = j1 + inter_only if not full else 0
        for j2 in range(j_int, n_features):
            temp = (X_cpt_Z[:, j2] * X_cpt_Z[:, j1] - mean_Z[jj]) / std_Z[jj]
            norm_Z = np.array(
                [norm_Z, np.abs(temp.T @ residuals - n_samples * alpha4 * theta[jj])]
            ).max()
            jj += 1
    norm_Z = (alpha / alpha2) * norm_Z

    cst_dual = max(alpha * n_samples, norm_X, norm_Z)
    nu = residuals / cst_dual

    new_dual = (
        y_norm2 / (2 * n_samples)
        - ((n_samples * alpha ** 2) / 2) * norm(nu - y / (alpha * n_samples)) ** 2
        - ((n_samples * alpha) / cst_dual) ** 2 * (alpha3) / 2 * norm(beta) ** 2
        - ((n_samples * alpha) / cst_dual) ** 2 * (alpha4) / 2 * norm(theta) ** 2
    )
    if new_dual > dual:
        dual = new_dual
    gap -= dual
    return (primal, dual, gap, nu, cst_dual)
