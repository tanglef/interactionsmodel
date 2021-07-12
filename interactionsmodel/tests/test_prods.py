import torch
import numpy as np
import pytest
from interactionsmodel.utils import cpt_mean_std
import interactionsmodel.utils.products as prod_utils
import benchmarks.bench_products as bench_fct
from interactionsmodel.utils import power_method, make_Z_full, make_Z, Lanczos

##############################
# Prepare everything we need
##############################

dtype = torch.float64
device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(11235813)
torch.manual_seed(11235813)

n = 50
p = 30
eps = 1e-4
q = int(p * (p + 1) / 2)
qfull = int(p ** 2)

X_np = np.random.randn(n, p).astype("float64")
XT_np = X_np.T.astype("float64")
X_tch = torch.from_numpy(X_np).to(device)
XT_tch = X_tch.T.contiguous()

theta_np = np.random.randn(q).astype("float64")
theta_tch = torch.from_numpy(theta_np).view(-1, 1).to(device)
theta_np_full = np.random.randn(qfull).astype("float64")
theta_tch_full = torch.from_numpy(theta_np_full).view(-1, 1).to(device)

beta_np = np.random.randn(n).astype("float64")
beta_tch = torch.from_numpy(beta_np).view(-1, 1).to(device)
beta_np_full = np.random.randn(n).astype("float64")
beta_tch_full = torch.from_numpy(beta_np_full).view(-1, 1).to(device)

Z_np = make_Z(X_np, bind="numpy").astype("float64")
Z_tch = torch.from_numpy(Z_np).to(device)
Z_tch_full = make_Z_full(X_tch)
Z_np_full = Z_tch_full.cpu().numpy()

meansX, stdX, meansZ, stdZ = cpt_mean_std(X_np)
res_np = np.zeros((n, 1)).astype("float64")
res_tch = torch.zeros((n, 1), device=device, dtype=dtype)
res_np_T = np.zeros((q, 1)).astype("float64")
res_tch_T = torch.zeros((q, 1), device=device, dtype=dtype)


####################################
# Tests the products from benchmark
# ----------------------------------
# We need to test them separatly because of the res
# argument (to be fair with the benchmark of numba)


# @pytest.mark.parametrize("res,X,XT,theta,fct", [
#  (res_tch.clone(), X_tch, XT_tch, theta_tch, bench_fct.product_Z_keops),
#  (res_tch.clone(), X_tch, XT_tch, theta_tch, bench_fct.product_Z_torch)
#  ])
# def test_bench_matvecZ(res, X, XT, theta, fct):
#     res_np = Z_np @ theta_np
#     res_fct = fct(X, XT, theta, res).reshape(-1).cpu()
#     assert np.allclose(res_np.ravel(), res_fct, eps)


# @pytest.mark.parametrize("res,X,XT,theta,fct", [
#     (res_tch_T.clone(), X_tch, XT_tch, beta_tch, bench_fct.product_ZT_keops),
#     (res_tch_T.clone(), X_tch, XT_tch, beta_tch, bench_fct.product_ZT_torch)
#     ])
# def test_bench_matvecZT(res, X, XT, theta, fct):
#     res_np = Z_np.T @ beta_np
#     res_fct = fct(X, XT, theta, res).reshape(-1).cpu()
#     assert np.allclose(res_np.ravel(), res_fct, eps)


# def test_bench_matvecZ_full():
#     res_np = Z_np_full @ theta_np_full
#     fct = bench_fct.product_Z_full
#     int_ = torch.zeros((n, 1), device=device, dtype=dtype)
#     res = fct(X_tch, XT_tch, theta_tch_full, int_).reshape(-1).cpu()
#     assert np.allclose(res_np.ravel(), res, eps)


# def test_bench_matvecZT_full():
#     res_np = Z_np_full.T @ beta_np_full
#     fct = bench_fct.product_ZT_full
#     int_ = torch.zeros((qfull, 1), device=device, dtype=dtype)
#     res = fct(X_tch, XT_tch, beta_tch_full, int_).reshape(-1).cpu()
#     assert np.allclose(res_np.ravel(), res, eps)


###################################
# Test preprocessing means-std
# ---------------------------------


def test_means_std():
    mX, sX, mZ, sZ = cpt_mean_std(X_tch, full=False, bind="torch")
    all_good = []
    all_good.append(np.allclose(meansX, mX.cpu()))
    all_good.append(np.allclose(stdX, sX.cpu()))
    obegin = 0
    for var in range(p):
        p_tilde = p - var
        assert np.allclose(meansZ[obegin : (obegin + p_tilde)], mZ[var].cpu())
        assert np.allclose(
            stdZ[obegin : (obegin + p_tilde)],
            sZ[var].cpu(),
        )
        obegin += p_tilde


####################################
# Tests the products from products
# ----------------------------------
# mean and std were tested so ok


mX, sX, mZ, sZ = cpt_mean_std(X_tch, full=False, bind="torch")
_, _, mZ_full, sZ_full = cpt_mean_std(X_tch, full=True, bind="torch")
means_Z_full = np.mean(Z_np_full, 0)
std_Z_full = np.std(Z_np_full, 0)


def test_utils_Z():
    res_np = (Z_np - meansZ) / stdZ @ theta_np
    res = prod_utils.product_Z(X_tch, theta_tch, mZ, sZ)
    assert np.allclose(res_np.ravel(), res.view(-1).cpu())


def test_utils_ZT():
    res_np = ((Z_np - meansZ) / stdZ).T @ beta_np
    res = prod_utils.product_ZT(X_tch, beta_tch, mZ, sZ)
    assert np.allclose(res_np.ravel(), res.view(-1).cpu())


def test_utils_Z_full():
    res_np = (Z_np_full - means_Z_full) / std_Z_full @ theta_np_full
    res = prod_utils.product_Z_full(X_tch, theta_tch_full, mZ_full, sZ_full)
    assert np.allclose(res_np.ravel(), res.view(-1).cpu())


def test_utils_ZT_full():
    res_np = ((Z_np_full - means_Z_full) / std_Z_full).T @ beta_np_full
    res = prod_utils.product_ZT_full(X_tch, beta_tch_full, mZ_full, sZ_full)
    assert np.allclose(res_np.ravel(), res.view(-1).cpu())


###########################
# Power method
# -------------------------


@pytest.mark.parametrize(
    "mat,which",
    [
        (X_tch, "X"),
        (
            (Z_tch - torch.from_numpy(meansZ).to(device))
            / torch.from_numpy(stdZ).to(device),
            "Z",
        ),
    ],
)
def test_power_method(mat, which):
    eigen = torch.linalg.norm(mat, 2) ** 2
    eigen_pgd = power_method(X_tch, which, mZ, sZ, eps=1e-9)
    assert np.allclose(eigen.cpu(), eigen_pgd.cpu())


@pytest.mark.parametrize(
    "mat,which",
    [
        (X_tch, "X"),
        (
            (Z_tch_full - torch.from_numpy(means_Z_full).to(device))
            / torch.from_numpy(std_Z_full).to(device),
            "Z",
        ),
    ],
)
def test_power_method_full(mat, which):
    eigen = torch.linalg.norm(mat, 2) ** 2
    eigen_pgd = power_method(X_tch, which, mZ_full, sZ_full, eps=1e-9, full=True)
    assert np.allclose(eigen.cpu(), eigen_pgd.cpu())


@pytest.mark.parametrize(
    "mat,which",
    [
        (X_tch, "X"),
        (
            (Z_tch - torch.from_numpy(meansZ).to(device))
            / torch.from_numpy(stdZ).to(device),
            "Z",
        ),
    ],
)
def test_lanczos(mat, which):
    eigen = torch.linalg.norm(mat, 2) ** 2
    eigen_pgd = Lanczos(X_tch, which, mZ, sZ, n_cv=100)
    assert np.allclose(eigen.cpu(), eigen_pgd.cpu())
