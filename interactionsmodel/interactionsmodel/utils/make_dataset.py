import numpy as np
from numpy.random import randint, randn
from numpy.random import choice
from interactionsmodel.utils import cpt_mean_std, whitening
from interactionsmodel.utils import cpt_lambda_max
from numba import njit


def make_data(
    X,
    inter_only,
    sigma,
    beta_sparsity,
    theta_sparsity,
    choice_features=np.array([-1, 1]),
    fit_interaction=True,
    bind="numpy",
    seed=11235813,
    full=False,
):
    """Create a dataset for interactions model

    Args:
        X (array): original data
        inter_only (bool): do not take squared interactions
        sigma (float): scale of noise
        beta_sparsity (int): sparsity in vector beta
        theta_sparsity (int): sparsity in vector theta
        choice_features (array, optional): values in where to chose
             for beta and theta. Defaults to np.array([-1, 1]).
        fit_interaction (bool, optional): Should take interactions.
             Defaults to True.
        bind (str, optional): use numpy of torch. Defaults to "numpy".
        seed (int, optional): Defaults to 11235813.

    Returns:
        y (array): X @ beta + Z @ theta + noise
        beta (array): created beta
        theta (array): created theta
        sigma (float): scale of the noise inputed
        noise (array): noise created
    """
    np.random.seed(seed)
    if bind != "torch":
        return make_data_np(
            X,
            inter_only,
            sigma,
            beta_sparsity,
            theta_sparsity,
            choice_features,
            fit_interaction,
            full,
        )
    else:
        raise NotImplementedError


def make_data_np(
    X,
    inter_only,
    sigma,
    beta_sparsity,
    theta_sparsity,
    choice_features=np.array([-1, 1]).astype("float64"),
    fit_interaction=True,
    full=False,
):
    n_samples, n_features = X.shape
    if inter_only:
        n_squarefeatures = int(n_features * (n_features - 1) / 2)
    elif not full:
        n_squarefeatures = int(n_features + n_features * (n_features - 1) / 2)
    else:
        n_squarefeatures = int(n_features ** 2)

    # Error size of parameter
    if beta_sparsity > n_features or theta_sparsity > n_squarefeatures:
        raise ValueError(
            "The number of active features of beta or theta can't be"
            "superior than the size of beta or theta"
        )

    # generate beta and theta according to parameter
    beta = np.zeros(n_features).astype(X.dtype)
    theta = np.zeros(n_squarefeatures).astype(X.dtype)

    idx_beta = np.sort(choice(np.arange(n_features), size=beta_sparsity, replace=False))
    beta[idx_beta] = choice(choice_features, beta_sparsity)
    idx_theta = np.sort(
        choice(np.arange(n_squarefeatures), size=theta_sparsity, replace=False)
    )
    theta[idx_theta] = choice(choice_features, theta_sparsity)

    mean_X, std_X, mean_Z, std_Z = cpt_mean_std(
        X, inter_only, fit_interaction, full=full
    )

    noise = np.random.normal(loc=0, scale=sigma, size=n_samples).astype(X.dtype)

    y = np.zeros(n_samples).astype(X.dtype)
    jj = 0
    for j1 in range(n_features):
        X_tmp = (X[:, j1] - mean_X[j1]) / std_X[j1]
        y += X_tmp * beta[j1]
    for j1 in range(n_features):
        j_int = j1 + inter_only if not full else 0
        for j2 in range(j_int, n_features):
            Z_tmp = X[:, j2] * X[:, j1]
            Z_tmp = (Z_tmp - mean_Z[jj]) / std_Z[jj]
            y += Z_tmp * theta[jj]
            jj += 1
    y += noise

    return (y, beta, theta, sigma, noise)


def get_lambda_max(
    X,
    y,
    X_cpt_Z=None,
    mean_X=None,
    mean_Z=None,
    std_X=None,
    std_Z=None,
    l1_ratio=1,
    bind="numpy",
    zca=False,
    eps=0.0,
):
    if X_cpt_Z is None:
        X_cpt_Z = X.copy() if bind != "torch" else X.clone()
        mean_X, std_X, mean_Z, std_Z = cpt_mean_std(X, False, True, True, bind=bind)
    if zca:
        Xt = X - mean_X
        mat_zca = whitening(Xt, eps=eps, bind=bind)
        _, _, mean_Z, std_Z = cpt_mean_std(Xt @ mat_zca, False, True, False, bind=bind)
        zca = mat_zca
    else:
        zca = False
    if bind == "torch":
        if zca is False:
            Xt = X - mean_X
    X_ = X if zca is False else Xt @ mat_zca
    lambda1_max, lambda2_max = cpt_lambda_max(
        X_,
        X_cpt_Z,
        y,
        mean_X,
        mean_Z,
        std_X,
        std_Z,
        l1_ratio,
        False,
        True,
        True,
        True,
        bind=bind,
        zca=zca,
    )
    return lambda1_max, lambda2_max


@njit
def make_regression(
    n_samples,
    n_features,
    fit_intercept=True,
    mu=0,
    sigma=1,
    inter_only=False,
    sparse_main=0.25,
    sparse_inter=0.25,
    seed=21071994,
    snr=None,
    ltheta=-(10 ** 2),
    utheta=10 ** 2,
    lbeta=-(10 ** 3),
    ubeta=10 ** 3,
    X=None,
):
    """
    This function have for aim to randomly generate random data
    with interaction features. There includes a sparsity parameter
    for the beta AND the theta. This function does not permit to
    control the level of sparsity for the main effect and the interaction
    independantly.

    Input :
    n_samples : a numeric, the number of samples.
    n_features : a numeric, the number of features.
    inter_only : a boolean, control if we include or not the
    quadratic effect. Default is : False (quadratic effect are include).
    sparse : a numerirc, control the level of sparisity.
    seed : a numeric, the seed of the random generator.
    snr: signal to noise ratio (overrides sigma)

    Output :
    X : 2D array (numeric), a randomly generate  design matrix
    y : 1D array (numeric), a response vector : y = X@beta + Z@theta + noise,
    where noise is 1D array generate from a Gaussian distribution.
    beta : 1D array (numeric), a random sparse vector to
    the main effect.
    theta : 1D array (numeric), a random sparse vector associated
    to the interaction effect.
    """
    np.random.seed(seed)
    if X == None:
        X = mu + sigma * randn(n_samples * n_features).reshape((n_samples, n_features))
    if fit_intercept:
        beta0 = randint(low=-(10 ** 1), high=10 ** 1, size=1)[0]
    else:
        beta0 = 0
    y = beta0 * np.ones(n_samples)

    beta = np.zeros(shape=n_features)
    beta_sparse = int(n_features * sparse_main)
    choice_array = np.arange(n_features)
    beta_indx = np.random.choice(choice_array, size=beta_sparse, replace=False)
    beta_value = randint(low=lbeta, high=ubeta, size=beta_sparse)
    beta[beta_indx] = beta_value

    if inter_only:
        n_squarefeature = int(n_features * (n_features - 1) / 2)
    else:
        n_squarefeature = int(n_features + n_features * (n_features - 1) / 2)

    theta_sparse = int(n_squarefeature * sparse_inter)
    theta = np.zeros(shape=n_squarefeature)
    choice_array = np.arange(n_squarefeature)
    theta_indx = np.random.choice(choice_array, size=theta_sparse, replace=False)
    theta_value = randint(low=ltheta, high=utheta, size=theta_sparse)
    theta[theta_indx] = theta_value

    Z = np.zeros(n_squarefeature)
    for i in range(n_samples):
        jj = 0
        for j1 in range(n_features):
            if inter_only:
                j_int = j1 + 1
            else:
                j_int = j1
            for j2 in range(j_int, n_features):
                Z[jj] = X[i, j1] * X[i, j2]
                jj += 1
        y[i] += X[i, :] @ beta + Z @ theta
    var_y = np.var(y)
    if snr is None:
        error = randn(n_samples)
        y += error
    else:
        var_error = var_y / snr
        error = np.sqrt(var_error) * randn(n_samples)
        y += error
    return (X, y, beta0, beta, theta)
