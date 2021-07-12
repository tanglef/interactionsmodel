import numpy as np
from numpy.linalg import norm
from numba import njit


@njit()
def cpt_norm(
    X,
    X_cpt_Z,
    mean_Z,
    std_Z,
    n_features,
    n_squarefeatures,
    fit_interaction=True,
    inter_only=False,
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
    """
    dtype = X.dtype
    X_norm2 = np.zeros(n_features).astype(dtype)
    Z_norm2 = np.zeros(n_squarefeatures).astype(dtype)

    jj = 0
    for j1 in range(n_features):
        # X_tmp = (X[:, j1] - mean_X[j1])/std_X[j1]
        X_norm2[j1] = norm(X[:, j1]) ** 2
    X_cpt_Z = X if zca else X_cpt_Z
    if fit_interaction:
        for j1 in range(n_features):
            j_int = j1 + inter_only if not full else 0
            for j2 in range(j_int, n_features):
                Z_tmp = X_cpt_Z[:, j2] * X_cpt_Z[:, j1]
                Z_tmp = (Z_tmp - mean_Z[jj]) / std_Z[jj]
                Z_norm2[jj] = norm(Z_tmp) ** 2
                jj += 1
    return (X_norm2, Z_norm2)


@njit()
def cpt_mean_std(
    X,
    inter_only=False,
    fit_interaction=True,
    standardize=True,
    inter_thn_std=True,
    full=False,
):

    # initialization of length variable
    _, n_features = X.shape
    if inter_only:
        n_squarefeature = int(n_features * (n_features - 1) / 2)
    elif not full:
        n_squarefeature = int(n_features + n_features * (n_features - 1) / 2)
    else:
        n_squarefeature = int(n_features ** 2)

    mean_X = np.zeros(n_features).astype(X.dtype)
    std_X = np.ones(n_features).astype(X.dtype)
    for j1 in range(n_features):
        mean_X[j1] = np.mean(X[:, j1])
        std_X[j1] = np.std(X[:, j1])

    mean_Z = np.zeros(n_squarefeature).astype(X.dtype)
    std_Z = np.ones(n_squarefeature).astype(X.dtype)

    if fit_interaction:
        if standardize:
            if inter_thn_std:
                mean_X_tmp = np.zeros(n_features).astype(X.dtype)
                std_X_tmp = np.ones(n_features).astype(X.dtype)
            else:
                mean_X_tmp = mean_X
                std_X_tmp = std_X
        else:  # fit_intercept case
            if inter_thn_std:
                mean_X_tmp = np.zeros(n_features).astype(X.dtype)
            else:
                mean_X_tmp = mean_X
            std_X_tmp = np.ones(n_features).astype(X.dtype)

        jj = 0
        for j1 in range(n_features):
            j_int = j1 + inter_only if not full else 0
            for j2 in range(j_int, n_features):
                Z_tmp = (
                    (X[:, j2] - mean_X_tmp[j2])
                    / std_X_tmp[j2]
                    * (X[:, j1] - mean_X_tmp[j1])
                    / std_X_tmp[j1]
                )
                mean_Z[jj] = np.mean(Z_tmp)
                std_Z[jj] = np.std(Z_tmp)
                jj += 1
    return (mean_X, std_X, mean_Z, std_Z)


@njit()
def cpt_alpha_max(
    X,
    y,
    mean_X,
    mean_Z,
    std_X,
    std_Z,
    l1_ratio=1,
    inter_only=False,
    fit_intercept=True,
    fit_interaction=True,
    standardize=False,
    zca=False,
):
    """
    Compute the parameters alpha_max1 and alpha_max2
    associated to the L1 norm s.t. for this alpha_max1
    (resp. alpha_max2), the main (resp. interaction) effects
    are null.

    Output :
    alpha_max1 : the parameter associated to the L1 norm of
    the main effect, such that beta is null.
    alpha_max2 : the parameter associated to the L1 norm of
    the interaction effect, such that theta is null.
    """
    n_samples, n_features = X.shape
    Xtype = X.dtype
    if inter_only:
        n_squarefeature = int(n_features * (n_features - 1) / 2)
    else:
        n_squarefeature = int(n_features + n_features * (n_features - 1) / 2)

    # to avoid useless scalar produce, do it just one time
    if standardize:
        t_mean_X = n_features == len(mean_X)
        t_mean_Z = n_squarefeature == len(mean_Z)
        t_std_X = n_features == len(std_X)
        t_std_Z = n_squarefeature == len(std_Z)
        if (t_mean_X and t_mean_Z and t_std_X and t_std_Z) is False:
            raise ValueError("Dimension of means and std do not match for X and Z")
    elif fit_intercept:
        t_mean_X = n_features == len(mean_X)
        t_mean_Z = n_squarefeature == len(mean_Z)
        std_X = np.ones(n_features).astype(Xtype)
        std_Z = np.ones(n_squarefeature).astype(Xtype)
        if (t_mean_X and t_mean_Z) is False:
            raise ValueError("Dimension in mean_X and mean_Z do not match for X and Z")
    else:
        mean_X = np.zeros(n_features).astype(Xtype)
        std_X = np.ones(n_features).astype(Xtype)
        mean_Z = np.zeros(n_squarefeature).astype(Xtype)
        std_Z = np.ones(n_squarefeature).astype(Xtype)

    X_all_tmp = (X - mean_X) / std_X if zca is False else X  # X is already Xt @ zca
    alpha_max1 = np.max(np.abs(X_all_tmp.T @ y)) / (n_samples * l1_ratio)
    if fit_interaction:
        alpha_tmp = 0
        jj = 0
        for j1 in range(n_features):
            if inter_only:
                j_int = j1 + 1
            else:
                j_int = j1
            for j2 in range(j_int, n_features):
                Z_tmp = (X[:, j1] * X[:, j2] - mean_Z[jj]) / std_Z[jj]
                tmp = np.abs(Z_tmp.T @ y)
                alpha_tmp = np.max(np.array([tmp, alpha_tmp]))
                jj += 1
        alpha_max2 = alpha_tmp / (n_samples * l1_ratio)
    else:
        alpha_max2 = 0
    return (alpha_max1, alpha_max2)
