import numpy as np
import torch
from sklearn.linear_model import ElasticNet
from interactionsmodel.solvers import PGD, CD, CBPG_CS, CBPG_CS_mod, CBPG_permut
from interactionsmodel.utils.numba_fcts import cpt_mean_std, cpt_norm, cpt_alpha_max
from interactionsmodel.utils import make_Z, make_data
from numpy.random import multivariate_normal
from interactionsmodel.utils import kkt_violation


use_cuda = torch.cuda.is_available()
dtype = torch.float64
device = "cuda" if use_cuda else "cpu"
MAXITER = int(5e2)
tol = 1e-6
seed = 11235813
np.random.seed(seed)

n_samples, n_features = 100, 5
inter_only = False
snr, corr = 10, 0.9
beta_sparsity, theta_sparsity = 5, 5
choice_features = np.array([-10, 10]).astype("float64")
corr_expected = np.zeros((n_features, n_features)).astype("float64")
for a in range(n_features):
    for i in range(n_features):
        corr_expected[a, i] = corr ** abs(a - i)

X = multivariate_normal(
    mean=np.arange(n_features), cov=corr_expected, size=n_samples
).astype("float64")
X_cpt_Z = X.copy(order="F")
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

Z = make_Z(X_cpt_Z, bind="np").astype("float64")

mean_X, std_X, mean_Z, std_Z = cpt_mean_std(X)
W = np.hstack([(X - mean_X) / std_X, (Z - mean_Z) / std_Z])
X = X.copy(order="F")
mean_y = np.mean(y)

n_squarefeatures = int(n_features * (n_features + 1) / 2)
standardize = False
fit_intercept = False
fit_interaction = True
l1_ratio = 0.95

X_norm2, Z_norm2 = cpt_norm(
    X, X_cpt_Z, mean_Z, std_Z, n_features, n_squarefeatures, fit_interaction
)

alpha1, alpha2 = cpt_alpha_max(
    X,
    y,
    mean_X=mean_X,
    mean_Z=mean_Z,
    std_X=std_X,
    std_Z=std_Z,
    l1_ratio=l1_ratio,
    standardize=standardize,
    fit_intercept=fit_intercept,
)
alphamax = max(alpha1, alpha2)
alpha = alphamax / 10
alpha1 /= 10
alpha2 /= 10
alphas = np.array(
    [alpha * l1_ratio, alpha * l1_ratio, alpha * (1 - l1_ratio), alpha * (1 - l1_ratio)]
)

result_skl = ElasticNet(
    alpha=alpha,
    l1_ratio=l1_ratio,
    fit_intercept=fit_intercept,
    max_iter=MAXITER,
    tol=tol,
)
result_skl.fit(W, y)
beta_skl = result_skl.coef_


def test_sklearn_enet_int_cd():
    cd = CD(X, y, alphas)
    cd.run(MAXITER, eps=tol)
    beta = cd.beta
    theta = cd.theta
    assert np.allclose(beta_skl, np.hstack([beta, theta]), rtol=1e-4)


X_tch = torch.from_numpy(X).to(device)
y_tch = torch.from_numpy(y).view(-1, 1).to(device)
alphas_tch = torch.from_numpy(alphas).to(device)


def test_sklearn_pgd():
    pgd = PGD(X_tch, y_tch, alphas_tch, device)
    pgd.run(MAXITER, eps=tol)
    beta = pgd.beta
    theta = pgd.theta
    stack = np.hstack([beta.flatten().cpu().numpy(), theta.flatten().cpu().numpy()])
    assert np.allclose(beta_skl, stack)


def test_sklearn_cbpg_cs():
    cbpg = CBPG_CS(X_tch, y_tch, alphas_tch, device)
    cbpg.run(MAXITER, eps=tol)
    beta = cbpg.beta
    theta = cbpg.theta
    stack = np.hstack([beta.flatten().cpu().numpy(), theta.flatten().cpu().numpy()])
    assert np.allclose(beta_skl, stack)


def test_sklearn_rando_cbpg():
    cbpg_rand = CBPG_CS_mod(X_tch, y_tch, alphas_tch, device)
    cbpg_rand.run(MAXITER, eps=tol)
    beta = cbpg_rand.beta
    theta = cbpg_rand.theta
    stack = np.hstack([beta.flatten().cpu().numpy(), theta.flatten().cpu().numpy()])
    assert np.allclose(beta_skl, stack)


def test_sklearn_permut_cbpg():
    cbpg_per = CBPG_permut(X_tch, y_tch, alphas_tch, device)
    cbpg_per.run(MAXITER, eps=tol)
    beta = cbpg_per.beta
    theta = cbpg_per.theta
    stack = np.hstack([beta.flatten().cpu().numpy(), theta.flatten().cpu().numpy()])
    assert np.allclose(beta_skl, stack)


##########################
# Test duality gap
##########################
# init all to avoid running them again and again
# Classic CBPG isn't used as the Lipschitz constant simply do not allow cv


pgd = PGD(X_tch, y_tch, alphas_tch, device)
pgd.run(MAXITER, eps=tol)
cd = CD(X, y, alphas)
cd.run(MAXITER, eps=tol)
cbpg_cs = CBPG_CS(X_tch, y_tch, alphas_tch, device)
cbpg_cs.run(MAXITER, eps=tol)
f_pgd = PGD(X_tch, y_tch, alphas_tch, device, full=True)
f_pgd.run(MAXITER, eps=tol)
f_cbpg = CBPG_CS(X_tch, y_tch, alphas_tch, device, full=True)
f_cbpg.run(MAXITER, eps=tol)
rando_cbpg = CBPG_CS_mod(X_tch, y_tch, alphas_tch, device)
rando_cbpg.run(MAXITER, eps=tol)
cbpg_per = CBPG_permut(X_tch, y_tch, alphas_tch, device)
cbpg_per.run(MAXITER, eps=tol)


def test_kkt_viol():
    max_viol = 1e-6
    kkt_cd = kkt_violation(cd, cd.beta, cd.theta, bind="np")
    assert kkt_cd < max_viol

    kkt_pgd = kkt_violation(pgd, pgd.beta, pgd.theta, bind="torch")
    assert np.isclose(kkt_pgd.cpu().type(torch.float), kkt_cd)

    kkt_cbpgcs = kkt_violation(cbpg_cs, cbpg_cs.beta, cbpg_cs.theta, bind="torch")
    assert np.isclose(kkt_cbpgcs.cpu().type(torch.float), kkt_cd)

    kkt_per = kkt_violation(cbpg_per, cbpg_per.beta, cbpg_per.theta, bind="torch")
    assert np.isclose(kkt_per.cpu().type(torch.float), kkt_cd)


def test_duality_gaps():
    max_gap = 2e-3
    gap_cd = cd.get_dual_gap()[2]
    assert gap_cd < max_gap

    gap_pgd = pgd.get_dual_gap()[2].cpu()
    assert np.isclose(gap_pgd, gap_cd)

    gap_cbpg_cs = cbpg_cs.get_dual_gap()[2].cpu()
    assert np.isclose(gap_cbpg_cs, gap_cd)

    gap_f_pgd = f_pgd.get_dual_gap()[2].cpu()
    assert np.isclose(gap_f_pgd, gap_cd)

    gap_f_cbpg = f_cbpg.get_dual_gap()[2].cpu()
    assert np.isclose(gap_f_cbpg, gap_cd)

    gap_rando_cbpg = rando_cbpg.get_dual_gap()[2].cpu()
    assert np.isclose(gap_rando_cbpg, gap_cd)

    gap_per = cbpg_per.get_dual_gap()[2].cpu()
    assert np.isclose(gap_per, gap_cd)


def test_objectives():
    true_obj = cd.get_objective()
    pgd_obg = pgd.get_objective()
    assert np.allclose(true_obj, pgd_obg.cpu())

    cbpg_obj = cbpg_cs.get_objective()
    assert np.allclose(true_obj, cbpg_obj.cpu())

    rando_obj = rando_cbpg.get_objective()
    assert np.allclose(true_obj, rando_obj.cpu())

    per_obj = cbpg_per.get_objective()
    assert np.allclose(per_obj.cpu(), true_obj)


#############################
# Full solvers
#############################


def test_full_objectives():
    true_obj = cd.get_objective()
    f_pgd_obg = f_pgd.get_objective()
    assert np.allclose(true_obj, f_pgd_obg.cpu())

    f_cbpg_obj = f_cbpg.get_objective()
    assert np.allclose(true_obj, f_cbpg_obj.cpu())


def test_full_compo():
    beta_ref, theta_ref = cd.beta, cd.theta
    beta_fpgd, theta_fpgd = f_pgd.beta, f_pgd.theta
    beta_fcbpg, theta_fcbpg = f_cbpg.beta, f_cbpg.theta

    # check beta
    assert np.allclose(beta_ref, beta_fpgd.view(-1).cpu())
    assert np.allclose(beta_ref, beta_fcbpg.view(-1).cpu())
    theta_fpgd = theta_fpgd.view(n_features, n_features)
    theta_fcbpg = theta_fpgd.view(n_features, n_features)

    # check symmetry
    assert np.allclose(theta_fcbpg.cpu(), theta_fcbpg.T.cpu())
    assert np.allclose(theta_fpgd.cpu(), theta_fpgd.T.cpu())

    # check theta
    begin = 0
    for var in range(n_features):
        next_ = begin + n_features - var
        diag_fcbpg = theta_fcbpg[var, var]
        diag_fpgd = theta_fpgd[var, var]
        assert np.allclose(theta_ref[begin], diag_fcbpg.cpu())
        assert np.allclose(theta_ref[begin], diag_fpgd.cpu())
        xtradiag_fcbpg = theta_fcbpg[var, (var + 1) :].view(-1)
        xtradiag_fpgd = theta_fpgd[var, (var + 1) :].view(-1)
        assert np.allclose(
            0.5 * theta_ref[(begin + 1) : next_], xtradiag_fcbpg.cpu(), rtol=1e-4
        )
        assert np.allclose(
            0.5 * theta_ref[(begin + 1) : next_], xtradiag_fpgd.cpu(), rtol=1e-4
        )
        begin += n_features - var


##############################
# Accelerated versions
##############################

acc_pgd = PGD(X_tch, y_tch, alphas_tch, device, use_acceleration=True)
acc_pgd.run(MAXITER, eps=tol)
acc_cbpg = CBPG_CS(X_tch, y_tch, alphas_tch, device, use_acceleration=True)
acc_cbpg.run(MAXITER, eps=tol)
acc_rando = CBPG_CS_mod(X_tch, y_tch, alphas_tch, device, use_acceleration=True)
acc_rando.run(MAXITER, eps=tol)
# print(f"CBPG cv in {cbpg_cs.cv} and accelerated version in {acc_cbpg.cv}")
# print(f"PGD cv in {pgd.cv} and accelerated version in {acc_pgd.cv}")
acc_perm = CBPG_permut(X_tch, y_tch, alphas_tch, device, use_acceleration=True)
acc_perm.run(MAXITER, eps=tol)


def test_duality_gaps_acc():
    max_gap = 1e-5

    gap_pgd = acc_pgd.get_dual_gap()[2].cpu()
    assert gap_pgd < max_gap

    gap_cbpg = acc_cbpg.get_dual_gap()[2].cpu()
    assert np.isclose(gap_cbpg, gap_pgd)

    # gap_rando = acc_rando.get_dual_gap()[2].cpu()
    # assert np.isclose(gap_rando, gap_pgd)

    # gap_perm = acc_perm.get_dual_gap()[2].cpu()
    # assert np.isclose(gap_perm, gap_pgd)


def test_objectives_acc():
    true_obj = cd.get_objective()
    acc_pgd_obg = acc_pgd.get_objective()
    assert np.allclose(true_obj, acc_pgd_obg.cpu())

    acc_cbpg_obj = acc_cbpg.get_objective()
    assert np.allclose(true_obj, acc_cbpg_obj.cpu())

    acc_rando_obj = acc_rando.get_objective()
    assert np.allclose(true_obj, acc_rando_obj.cpu())

    acc_perm_obj = acc_perm.get_objective().cpu()
    assert np.allclose(acc_perm_obj, true_obj)
