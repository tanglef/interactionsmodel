"""
=========================
Proximal GRADIENT DESCENT
=========================

Method
    ├── preprocessing:
    |   ├── standardize X
    |   └── compute means and std for Z (full or not)
    |
    └── PGD method:
        └── at step 0<=k<=K:
            ├── β^{k+1} <- prox(β^k - 1/LX * ∇_β f(β^k, ϴ^k))
            |      prox being with constant 1/LX over g^1
            └── ϴ^{k+1} <- prox(ϴ^k - 1/LZ * ∇_ϴ f(β^k, ϴ^k))
                   prox being with constant 1/LZ over g^2

Lipschitz constants computed w/ power iteration.
Nesterov acceleration available:
    -
"""

import torch
from interactionsmodel.utils import (
    PRODS,
    Lanczos,
    cpt_mean_std,
    dual_gap_enet,
    whitening,
    kkt_violation_tch,
)
from math import sqrt


class PGD:
    def __init__(
        self,
        X,
        y,
        alphas,
        device,
        full=False,
        use_acceleration=False,
        benchmark=True,
        zca=False,
        **kwargs
    ):
        """Proximal Gradient Descent solver.

        Args:
            X (tensor): original data of size (n,p)
            y (tensor): response variable of size (n,1)
            alphas (tensor): the four penalties (2 l1, 2 l2)
            device (string): typically 'cuda' or 'cpu'
            full (bool, optional): use p^2 interactions. Defaults to False.
            use_acceleration (bool, optional): use Nesterov acceleration.
                Defaults to False.
            benchmark (bool, optional): Should the callback NOT be timed.
                Defaults to True.
        """
        self.X_ = X.clone().contiguous().to(device)
        self.X = X.clone().contiguous().to(device)  # we don't modify original
        self.y = y.clone().contiguous().to(device)
        self.alphas = alphas
        self.device = device
        self.n, self.p = X.shape
        prods = PRODS["full"] if full else PRODS["small"]
        self.prod_Z = prods[0]
        self.benchmark = benchmark
        self.prod_ZT = prods[1]
        self.full = full
        self.meanX, self.stdX, self.meanZ, self.stdZ = cpt_mean_std(
            X, full=self.full, bind="torch"
        )
        self.X -= self.meanX
        self.zca = zca
        if zca:
            self.mat_zca = whitening(X, eps=0.0, bind="torch")
            self.X = self.X @ self.mat_zca
            self.X_ = self.X.clone()
            _, _, self.meanZ, self.stdZ = cpt_mean_std(
                self.X, full=False, standardize=False, bind="torch"
            )
        else:
            self.X /= self.stdX
        self.use_acceleration = use_acceleration
        if self.full:  # accelerate get_objective in the benchmarks
            _, _, self.meanZ_sm, self.stdZ_sm = cpt_mean_std(self.X_, bind="torch")
        else:
            self.meanZ_sm, self.stdZ_sm = self.meanZ, self.stdZ

    def run(
        self, maxiter, callback=None, Li=None, LX=None, LZ=None, eps=1e-4, **kwargs
    ):
        """Run the solver

        Args:
            maxiter (int): Maximal number of epochs
                (number of times each block is updated)
            callback (function, optional): criterion to stop the computation.
                Must return a boolean to indicate if the solver must continue.
                Defaults to None.
        """
        if callback is None:

            def callback(k, beta, theta):
                return k < maxiter

        if not self.full:
            Li, LX, LZ = self.run_pgd(maxiter, callback, LX=LX, LZ=LZ, eps=eps)
            return Li, LX, LZ
        else:
            self.run_full_pgd(maxiter, callback)

    def run_pgd(self, maxiter, callback=None, Li=None, LX=None, LZ=None, eps=1e-4):

        n, p = self.X.shape
        XT = self.X.T
        if LX is None or LZ is None:
            LX = Lanczos(self.X, which="X", meanZ=None, stdZ=None, n_cv=20) / n
            LZ = (
                Lanczos(self.X_, which="Z", meanZ=self.meanZ, stdZ=self.stdZ, n_cv=20)
                / n
            )

        n_interactions = int(p * (p + 1) / 2)
        beta = torch.zeros((p, 1), device=self.device, dtype=self.X.dtype)
        theta = torch.zeros((n_interactions, 1), device=self.device, dtype=self.X.dtype)
        resid = -self.y
        tnew = 1
        k = 0
        if self.benchmark:
            _ = callback(k, beta, theta)
        else:
            _ = callback(k, beta, theta, resid=resid)
        cond = True
        while cond:
            g_beta = 1 / n * (XT @ resid)
            updatable = beta - 1 / LX * g_beta
            nbeta = self.update_step(beta - 1 / LX * g_beta, 1 / LX)
            if self.use_acceleration:
                told = tnew
                tnew = (1 + sqrt(1 + 4 * told ** 2)) / 2
            diff_beta = nbeta - beta
            if torch.abs(diff_beta).sum() != 0:
                if self.use_acceleration:
                    nbeta += (told - 1) / tnew * diff_beta
                    resid = resid + self.X @ (nbeta - beta)
                else:
                    resid = resid + self.X @ diff_beta
            g_theta = (
                1
                / n
                * self.prod_ZT(
                    self.X_,
                    resid,
                    self.meanZ,
                    self.stdZ,
                )
            )
            updatable = theta - 1 / LZ * g_theta
            ntheta = self.update_step(updatable, 1 / LZ, which="theta")
            diff_theta = ntheta - theta
            if torch.abs(diff_theta).sum() != 0:
                if self.use_acceleration:
                    ntheta += (told - 1) / tnew * diff_theta
                    resid += self.prod_Z(self.X_, ntheta - theta, self.meanZ, self.stdZ)
                else:
                    resid += self.prod_Z(self.X_, diff_theta, self.meanZ, self.stdZ)
            if self.benchmark:
                cond = callback(k, nbeta, ntheta)
            else:
                cond = callback(k, nbeta, ntheta, resid)
            beta, theta = nbeta, ntheta
            kkt_ = kkt_violation_tch(self, beta, theta, resid)
            if kkt_ < eps:
                break
            k += 1
        self.beta = beta
        self.theta = theta
        if self.benchmark:
            callback(k - 1, self.beta, self.theta)
        else:
            callback(k - 1, self.beta, self.theta, resid=resid)

        self.cv = k
        return Li, LX, LZ

    def update_step(self, x, lambda_, which="beta"):
        which = 0 if which == "beta" else 1
        val = lambda_ * self.alphas[which]
        thresh = x - torch.clamp(x, -val, val)
        return 1 / (1 + lambda_ * self.alphas[which + 2]) * thresh

    def run_full_pgd(self, maxiter, callback=None):

        n, p = self.X.shape
        XT = self.X.T
        LX = torch.sqrt(Lanczos(self.X, which="X", meanZ=None, stdZ=None, n_cv=20)) / n
        LZ = (
            Lanczos(
                self.X_, which="Z", meanZ=self.meanZ, stdZ=self.stdZ, n_cv=20, full=True
            )
            / n
        )
        n_interactions = int(p ** 2)
        beta = torch.zeros((p, 1), dtype=self.X.dtype, device=self.device)
        theta = torch.zeros((n_interactions, 1), dtype=self.X.dtype, device=self.device)
        resid = -self.y
        k = 0
        if self.benchmark:
            _ = callback(k, beta, theta)
        else:
            _ = callback(k, beta, theta, resid=resid)
        cond = True
        while cond:
            g_beta = 1 / n * (XT @ resid)
            nbeta = self.update_step(beta - 1 / LX * g_beta, 1 / LX, which="beta")
            diff_beta = nbeta - beta
            if torch.abs(diff_beta).sum() != 0:
                resid = resid + self.X @ diff_beta
            g_theta = (
                1
                / n
                * self.prod_ZT(
                    self.X_,
                    resid,
                    self.meanZ,
                    self.stdZ,
                )
            )
            updatable = theta - 1 / LZ * g_theta
            ntheta = self.update_step_full(updatable, 1 / LZ)
            diff_theta = ntheta - theta

            if torch.abs(diff_theta).sum() != 0:
                resid += self.prod_Z(self.X_, diff_theta, self.meanZ, self.stdZ)
            if self.benchmark:
                cond = callback(k, nbeta, ntheta)
            else:
                cond = callback(k, nbeta, ntheta, resid)
            beta, theta = nbeta, ntheta
            k += 1
        self.beta = beta
        self.theta = theta
        if self.benchmark:
            callback(k - 1, self.beta, self.theta)
        else:
            callback(k - 1, self.beta, self.theta, resid=resid)

    def update_step_full(self, x, lambda_):  # lambda_ = 1/LZ
        p = self.X.shape[1]
        val = lambda_ * self.alphas[1]
        thresh_diag = 1 / (1 + lambda_ * self.alphas[3])
        thresh_out = 1 / (1 + 2 * lambda_ * self.alphas[3])
        thresh = x - torch.clamp(x, -val, val)
        begin = 0
        for var in range(p):
            next_ = begin + p
            thresh[begin : (begin + var)] *= thresh_out
            thresh[begin + var] *= thresh_diag
            thresh[(begin + var + 1) : (next_)] *= thresh_out
            begin += p
        return thresh

    def get_objective(self, beta=None, theta=None):
        """Compute the objective value"""
        if beta is None or theta is None:
            beta, theta = self.beta, self.theta
        n, p = self.X.shape
        if self.full:
            theta = theta.view(p, p)
            theta_chg = torch.zeros(
                (int(p * (p + 1) / 2), 1), dtype=self.X.dtype, device=self.device
            )
            obegin = 0
            for idx in range(p):
                p_tilde = p - idx
                theta_chg[obegin] = theta[idx, idx]
                extra_row = theta[idx, (idx + 1) :].view(-1, 1)
                theta_chg[(obegin + 1) : (obegin + p_tilde)] = 2 * extra_row
                obegin += p_tilde
            thetaprod = PRODS["small"][0](
                self.X_, theta_chg, self.meanZ_sm, self.stdZ_sm
            )
        else:
            theta_chg = theta
            thetaprod = self.prod_Z(self.X_, theta_chg, self.meanZ, self.stdZ)
        mc_int = self.y - self.X @ beta - thetaprod
        mc_int = mc_int.flatten()
        mc = 0.5 / n * (mc_int @ mc_int)
        beta_l1 = self.alphas[0] * torch.abs(beta).sum()
        beta_l2 = self.alphas[2] / 2 * (beta ** 2).sum()
        theta_l1 = self.alphas[1] * torch.abs(theta_chg).sum()
        theta_l2 = self.alphas[3] / 2 * torch.linalg.norm(theta_chg, 2) ** 2
        return mc + beta_l1 + beta_l2 + theta_l1 + theta_l2

    def get_dual_gap(self, beta=None, theta=None):
        if beta is None and theta is None:
            beta, theta = self.beta, self.theta
        n, p = self.X.shape
        if self.full:
            theta = theta.view(p, p)
            theta_chg = torch.zeros(
                (int(p * (p + 1) / 2), 1), dtype=self.X.dtype, device=self.device
            )
            obegin = 0
            for idx in range(p):
                p_tilde = p - idx
                theta_chg[obegin] = theta[idx, idx]
                extra_row = theta[idx, (idx + 1) :].view(-1, 1)
                theta_chg[(obegin + 1) : (obegin + p_tilde)] = 2 * extra_row
                obegin += p_tilde
            thetaprod = PRODS["small"][0](
                self.X_, theta_chg, self.meanZ_sm, self.stdZ_sm
            )
        else:
            theta_chg = theta
            thetaprod = self.prod_Z(self.X_, theta_chg, self.meanZ, self.stdZ)
        resid = self.y - self.X @ beta - thetaprod
        lbd_beta_l1, lbd_theta_l1, lbd_beta_l2, lbd_theta_l2 = self.alphas
        lambda_mean = (
            self.alphas[0] + self.alphas[1] + self.alphas[2] + self.alphas[3]
        ) / 4
        res = dual_gap_enet(
            beta,
            theta_chg,
            resid,
            lbd_beta_l1,
            lbd_theta_l1,
            lbd_beta_l2,
            lbd_theta_l2,
            lambda_mean,
            self.y,
            self.X,
            self.X_,
            self.meanZ_sm,
            self.stdZ_sm,
            n,
            p,
            torch.tensor([-1e6], dtype=self.X.dtype, device=self.device),
            torch.linalg.norm(self.y, 2) ** 2,
            bind="torch",
        )
        primal, dual, gap, nu, cst_dual = res
        return primal, dual, gap, nu, cst_dual
