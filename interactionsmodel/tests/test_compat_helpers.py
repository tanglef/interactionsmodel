from interactionsmodel.utils import (
    cpt_norm,
    cpt_mean_std,
    dual_gap_enet,
    cpt_lambda_max,
    whitening,
)
from interactionsmodel.utils import make_data
from interactionsmodel.utils import get_lambda_max
from numpy.random import multivariate_normal
import numpy as np
import torch
import pytest


seed = 11235813
n_samples, n_features = 100, 5
inter_only = False
snr, corr = 10, 0.9
beta_sparsity, theta_sparsity = 5, 5
choice_features = np.array([-10, 10])
corr_expected = np.zeros((n_features, n_features))
for a in range(n_features):
    for i in range(n_features):
        corr_expected[a, i] = corr ** abs(a - i)

X = multivariate_normal(mean=np.arange(n_features), cov=corr_expected, size=n_samples)
X_cpt_Z = np.copy(X, order="F")
(y, beta, theta, sigma, noise) = make_data(
    X,
    inter_only,
    1,  # scaled gaussian
    beta_sparsity,
    theta_sparsity,
    choice_features,
    True,
    seed=seed,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
X_tch = torch.from_numpy(X).to(device)
y_tch = torch.from_numpy(y).to(device)
beta_tch = torch.from_numpy(beta).to(device)
noise_tch = torch.from_numpy(noise).view(-1, 1).to(device)
theta_tch = torch.from_numpy(theta).to(device)
mean_X_np, std_X_np, mean_Z_np, std_Z_np = cpt_mean_std(X, bind="np")
mean_X_tc, std_X_tc, mean_Z_tc, std_Z_tc = cpt_mean_std(X_tch, bind="torch")
X_st_tc = (X_tch - mean_X_tc) / std_X_tc


def test_cpt_mean_std():
    all_close_meansX = np.allclose(mean_X_np, mean_X_tc.cpu())
    all_close_stdX = np.allclose(std_X_np, std_X_tc.cpu())
    meansZtc = []
    stdZtc = []
    for i in range(len(mean_Z_tc)):
        meansZtc.extend(mean_Z_tc[i].cpu())
        stdZtc.extend(std_Z_tc[i].cpu())
    all_close_meansZ = np.allclose(mean_Z_np, meansZtc)
    all_close_stdZ = np.allclose(stdZtc, std_Z_np)
    all_good = (
        all_close_meansX and all_close_stdX and all_close_meansZ and all_close_stdZ
    )
    assert all_good


def test_cpt_norm():
    p = X.shape[1]
    q = int(p * (p + 1) / 2)
    Xnorm2_np, Znorm2_np = cpt_norm(X, X_cpt_Z, mean_Z_np, std_Z_np, p, q, bind="numpy")
    Xnorm2_tc, Znorm2_tc = cpt_norm(
        X_tch, X_tch.clone(), mean_Z_tc, std_Z_tc, p, q, bind="torch", flatten=True
    )
    close_X = np.allclose(Xnorm2_np, Xnorm2_tc.cpu())
    Znorm2_tc = [val.cpu() for val in Znorm2_tc]
    close_Z = np.allclose(Znorm2_np, Znorm2_tc)
    assert close_X and close_Z


def test_lambda_max():
    lambda1_max_np, lambda2_max_np = cpt_lambda_max(
        X,
        X_cpt_Z,
        y,
        mean_X_np,
        mean_Z_np,
        std_X_np,
        std_Z_np,
        0.95,
        standardize=True,
        bind="numpy",
    )
    lambda1_max_tc, lambda2_max_tc = cpt_lambda_max(
        X_st_tc,
        X_tch.clone(),
        y_tch,
        mean_X_tc,
        mean_Z_tc,
        std_X_tc,
        std_Z_tc,
        0.95,
        bind="torch",
    )
    ok_lambda1 = np.allclose(lambda1_max_np, lambda1_max_tc.cpu())
    ok_lambda2 = np.allclose(lambda2_max_np, lambda2_max_tc.cpu())
    assert ok_lambda1 and ok_lambda2


lambda1_max_np, lambda2_max_np = cpt_lambda_max(
    X,
    X_cpt_Z,
    y,
    mean_X_np,
    mean_Z_np,
    std_X_np,
    std_Z_np,
    1,
    standardize=True,
    bind="numpy",
)
lambda1_max_tc, lambda2_max_tc = cpt_lambda_max(
    X_st_tc,
    X_tch.clone(),
    y_tch,
    mean_X_tc,
    mean_Z_tc,
    std_X_tc,
    std_Z_tc,
    1,
    bind="torch",
)


# @pytest.mark.parametrize("zca", [(True), (False)])
# def test_get_lambdas(zca):
#     lambda1, lambda2 = get_lambda_max(X_tch.clone(), y_tch,
#                                       bind="torch", zca=zca)
#     lambda1_max_np, lambda2_max_np = get_lambda_max(
#         X, y, X_cpt_Z, mean_X_np, mean_Z_np, std_X_np, std_Z_np,
#         1, bind="numpy", zca=zca
#     )
#     assert (np.allclose(lambda1.cpu(), lambda1_max_np) and
#             np.allclose(lambda2.cpu(), lambda2_max_np))


# @pytest.mark.parametrize("zca", [(True), (False)])
# def test_get_lambdas_direct(zca):
#     lambda1, lambda2 = get_lambda_max(X_tch.clone(), y_tch,
#                                       bind="torch", zca=zca)
#     lambda1_max_np, lambda2_max_np = get_lambda_max(
#         X, y, bind="numpy", zca=zca
#     )
#     assert (np.allclose(lambda1.cpu(), lambda1_max_np) and
#             np.allclose(lambda2.cpu(), lambda2_max_np))


def test_whitening():
    X_t = X_tch.clone() - mean_X_tc
    zca = whitening(X_t, eps=0.0, bind="torch")
    X_t = X_t @ zca
    zca_np = whitening(X - mean_X_np, eps=0.0, bind="numpy")
    X_tmp = (X - mean_X_np) @ zca_np
    close = np.allclose(zca.cpu(), zca_np)
    sig_tc = X_t.T @ X_t / X_t.shape[0]
    sig_np = X_tmp.T @ X_tmp / X_tmp.shape[0]
    assert (
        np.allclose(sig_tc.cpu(), np.eye(zca.shape[1]))
        and np.allclose(sig_np, np.eye(zca.shape[1]))
        and close
    )


def test_dual_gap():
    lambda1_np = np.array(lambda1_max_np) / 10
    lambda2_np = np.array(lambda2_max_np) / 10
    lambda1_tc = lambda1_max_tc / 10
    lambda2_tc = lambda2_max_tc / 10
    lambda_np = np.array(lambda1_np + lambda2_np) / 2
    lambda_tc = (lambda1_tc + lambda2_tc) / 2
    ynorm2_np = np.linalg.norm(y, 2) ** 2
    ynorm2_tc = torch.linalg.norm(y_tch, 2) ** 2
    dual = np.array([-1e6]).astype("float64")
    dual_tc = torch.from_numpy(dual).to(device)

    res_np = dual_gap_enet(
        beta,
        theta,
        noise,
        lambda1_np,
        lambda1_np,
        lambda2_np,
        lambda2_np,
        lambda_np,
        y,
        X,
        X_cpt_Z,
        mean_Z_np,
        std_Z_np,
        n_samples,
        n_features,
        dual[0],
        ynorm2_np,
    )
    res_tc = dual_gap_enet(
        beta_tch,
        theta_tch,
        noise_tch,
        lambda1_tc,
        lambda1_tc,
        lambda2_tc,
        lambda2_tc,
        lambda_tc,
        y_tch,
        X_tch,
        X_tch.clone(),
        mean_Z_tc,
        std_Z_tc,
        n_samples,
        n_features,
        dual_tc[0],
        ynorm2_tc,
        bind="torch",
    )
    all_good = [np.allclose(res_np[i], res_tc[i].cpu()) for i in range(len(res_np))]
    assert all_good
