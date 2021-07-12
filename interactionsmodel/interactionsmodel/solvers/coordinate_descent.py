import numpy as np
from numba import njit
from interactionsmodel.utils import cpt_mean_std, cpt_norm
from interactionsmodel.utils.numba_fcts import cd_enet_inter
from interactionsmodel.utils import dual_gap_enet
from interactionsmodel.utils import kkt_nb


@njit()
def run_second_part(
    alphas,
    X,
    y,
    y_norm2,
    Xnorm2,
    Znorm2,
    beta,
    theta,
    residuals,
    X_,
    mean_Z,
    std_Z,
    ynorm2,
    maxiter,
    eps,
    dtype,
    full,
):
    # tunning parameters
    alpha1 = alphas[0]
    alpha2 = alphas[1]
    alpha3 = alphas[2]
    alpha4 = alphas[3]
    n_samples, n_features = X.shape
    X_norm2_alpha_inv = 1 / (Xnorm2 + n_samples * alpha3)
    Z_norm2_alpha_inv = 1 / (Znorm2 + n_samples * alpha4)
    for itr in range(maxiter + 1):
        (beta, theta, residuals, beta_theta_max, d_beta_theta_max) = cd_enet_inter(
            beta,
            theta,
            residuals,
            False,
            alpha1,
            alpha2,
            X,
            X_,
            Xnorm2,
            X_norm2_alpha_inv,
            mean_Z,
            std_Z,
            Znorm2,
            Z_norm2_alpha_inv,
            n_samples,
            n_features,
            dtype,
            full,
        )
        kkt = kkt_nb(X, X_, y, mean_Z, std_Z, beta, theta, residuals, alphas)
        if abs(kkt) <= eps and itr > 10:
            break
    return beta, theta, itr


@njit()
def cpt_resid(X, X_, meanZ, stdZ, beta, theta, y):
    p = X.shape[1]
    y -= X @ beta
    jj = 0
    for j1 in range(p):
        for j2 in range(j1, p):
            Z_tmp = ((X_[:, j1] * X_[:, j2]) - meanZ[jj]) / stdZ[jj]
            y -= Z_tmp * theta[jj]
            jj += 1
    return y


class CD:
    def __init__(self, X, y, lambdas):
        self.X = X.copy(order="F")  # do not modify original
        self.y = y.copy(order="F")
        dtype = self.X.dtype
        self.dtype = dtype
        self.X_ = self.X.copy(order="F").astype(dtype)
        self.device = "cpu"  # match other solvers
        self.n, self.p = X.shape
        self.alphas = np.array(lambdas).astype(self.dtype).copy(order="F")
        self.full = False
        self.meanX, self.stdX, self.meanZ, self.stdZ = cpt_mean_std(
            self.X, full=self.full
        )
        self.X -= self.meanX.astype(dtype)
        self.X /= self.stdX.astype(dtype)

    # @profile  # noqa
    def run(
        self,
        maxiter,
        eps=1e-7,
        nb_breaks=None,
        callback=None,
        full=False,
        beta=None,
        theta=None,
        lambdas=None,
    ):
        _, p = self.X.shape

        q = int(p * (p + 1) / 2) if not self.full else int(p ** 2)
        if beta is None or theta is None:
            self.beta = np.zeros(p, order="F").astype(self.dtype)
            self.theta = np.zeros(q, order="F").astype(self.dtype)
            self.residuals = self.y.copy(order="F").astype(self.dtype)

        else:
            self.beta = beta.astype(self.dtype).copy(order="F")
            self.theta = theta.astype(self.dtype).copy(order="F")
            self.residuals = cpt_resid(
                self.X,
                self.X_,
                self.meanZ,
                self.stdZ,
                self.beta,
                self.theta,
                self.y.copy(order="F").astype(self.dtype),
            )
        if lambdas is not None:
            self.alphas = lambdas
        self.ynorm2 = (np.linalg.norm(self.y, 2) ** 2).astype(self.dtype)
        self.meany = np.mean(self.y).astype(self.dtype)
        self.Xnorm2, self.Znorm2 = cpt_norm(
            self.X, self.X_, self.meanZ, self.stdZ, p, q, full=self.full
        )
        self.Xnorm2 = self.Xnorm2.astype(self.dtype)
        self.Znorm2 = self.Znorm2.astype(self.dtype)
        # kkt0 = kkt_nb(self.X, self.X_, self.y.copy(order="F"), self.meanZ,
        #               self.stdZ,
        #               np.zeros(p, order='F').astype(self.dtype),
        #               np.zeros(q, order='F').astype(self.dtype),
        #               self.y.copy(order="F").astype(self.dtype),
        #               self.alphas)
        # eps *= kkt0
        self.beta, self.theta, self.cv = run_second_part(
            self.alphas,
            self.X,
            self.y,
            self.ynorm2,
            self.Xnorm2,
            self.Znorm2,
            self.beta,
            self.theta,
            self.residuals,
            self.X_,
            self.meanZ,
            self.stdZ,
            self.ynorm2,
            maxiter,
            eps,
            self.dtype,
            self.full,
        )
        if callback:
            callback(maxiter, self.beta, self.theta)

    def get_objective(self, beta=None, theta=None):
        """Compute the objective value"""
        if beta is None or theta is None:
            beta, theta = self.beta, self.theta
        n, p = self.X.shape
        theta_prod = np.zeros((n,)).astype(self.dtype)
        jj = 0
        for j1 in range(p):
            for j2 in range(j1, p):
                Z_tmp = (
                    (self.X_[:, j1] * self.X_[:, j2]) - self.meanZ[jj]
                ) / self.stdZ[jj]
                theta_prod += Z_tmp * theta[jj]
                jj += 1
        mc_int = self.y - self.X @ beta - theta_prod
        mc = 0.5 / n * (mc_int @ mc_int)
        beta_l1 = self.alphas[0] * np.abs(beta).sum()
        beta_l2 = self.alphas[2] / 2 * (beta ** 2).sum()
        theta_l1 = self.alphas[1] * np.abs(theta).sum()
        theta_l2 = self.alphas[3] / 2 * (theta ** 2).sum()
        return mc + beta_l1 + beta_l2 + theta_l1 + theta_l2

    def get_dual_gap(self, beta=None, theta=None):
        if beta is None and theta is None:
            beta, theta = self.beta, self.theta
        n, p = self.X.shape
        theta_prod = np.zeros((n,)).astype(self.dtype)
        jj = 0
        for j1 in range(p):
            j_int = j1 if not self.full else 0
            for j2 in range(j_int, p):
                Z_tmp = (
                    (self.X_[:, j_int] * self.X_[:, j2]) - self.meanZ[jj]
                ) / self.stdZ[jj]
                theta_prod += Z_tmp * theta[jj]
                jj += 1
        resid = self.y - self.X @ beta - theta_prod
        lbd_beta_l1, lbd_theta_l1, lbd_beta_l2, lbd_theta_l2 = self.alphas
        lambda_mean = (
            self.alphas[0] + self.alphas[1] + self.alphas[2] + self.alphas[3]
        ) / 4
        res = dual_gap_enet(
            beta,
            theta,
            resid,
            lbd_beta_l1,
            lbd_theta_l1,
            lbd_beta_l2,
            lbd_theta_l2,
            lambda_mean,
            self.y,
            self.X,
            self.X_,
            self.meanZ,
            self.stdZ,
            n,
            p,
            -1e6,
            np.linalg.norm(self.y, 2) ** 2,
            bind="numpy",
            full=self.full,
        )
        primal, dual, gap, nu, cst_dual = res
        return primal, dual, gap, nu, cst_dual
