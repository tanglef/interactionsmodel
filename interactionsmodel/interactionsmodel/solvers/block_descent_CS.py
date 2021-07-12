"""
=======================
BLOCK GRADIENT DESCENT
=======================

Method
    ├── preprocessing:
    |   ├── standardize X
    |   ├── compute means and std for Z (full or not)
    |   └── compute the lispchitz constant of each sub-block
    |
    └── CBPG method:
        └── at step 0<=k<=K:
            ├── β^{k+1} <- prox(β^k - 1/LX * ∇_β f(β^k, ϴ^k))
            |      prox being with constant 1/LX over g^1
            └── for i = 1,...,p, we use [i] for the i-th block
                └── ϴ^{k+1}[i] <- prox(ϴ^k[i] - 1/Li * ∇_ϴ[i] f(β^k, ϴ^k))
"""

import torch
from interactionsmodel.utils import PRODS, cpt_mean_std, dual_gap_enet, kkt_violation
from interactionsmodel.utils import Lanczos
from math import sqrt


class CBPG_CS:
    def __init__(
        self,
        X,
        y,
        alphas,
        device,
        full=False,
        use_acceleration=False,
        benchmark=True,
        **kwargs
    ):
        """Cyclic Block Proximal Gradient solver with lipschitz constants of
        each block upper-bounded using Cauchy-Schwarz.

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
        self.use_acceleration = use_acceleration
        self.device = device
        prods = PRODS["full"] if full else PRODS["small"]
        self.prod_Z = prods[0]
        self.prod_ZT = prods[1]
        self.full = full
        self.meanX, self.stdX, self.meanZ, self.stdZ = cpt_mean_std(
            X, full=self.full, bind="torch"
        )
        self.X -= self.meanX
        self.X /= self.stdX
        self.benchmark = benchmark
        self.n, self.p = X.shape
        self.XT = self.X.T.contiguous()

        if self.full:  # accelerate get_objective in the benchmarks
            _, _, self.meanZ_sm, self.stdZ_sm = cpt_mean_std(self.X_, bind="torch")
        else:
            self.meanZ_sm, self.stdZ_sm = self.meanZ, self.stdZ

    def run(
        self,
        maxiter,
        callback=None,
        Li=None,
        LX=None,
        eps=1e-4,
        beta=None,
        theta=None,
        recompute=False,
        alphas=None,
        **kwargs
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
                """Default callback only check the number of epochs."""
                return k <= maxiter

        if self.full:
            self.run_cbpg_full(maxiter, callback=callback)
        else:
            Li, LX, LZ = self.run_cbpg(
                maxiter,
                callback=callback,
                Li=Li,
                LX=LX,
                eps=eps,
                beta=beta,
                theta=theta,
                alphas=alphas,
                recompute=recompute,
            )
            return Li, LX, LZ

    def run_cbpg(
        self,
        maxiter,
        callback=None,
        Li=None,
        LX=None,
        LZ=None,
        eps=1e-4,
        beta=None,
        theta=None,
        recompute=False,
        alphas=None,
    ):
        """Run solver with q = p(p+1) / 2 interactions

        Args:
            maxiter (int): Maximum number of epochs
            callback (function, optional): criterion to stop the computation.
            Li (list, optional): list of all block Lipschitz constants
        """
        if alphas is not None:
            self.alphas = alphas
        n, p = self.n, self.p
        n_interactions = int(p * (p + 1) / 2)
        if beta is None or theta is None:
            beta = torch.zeros((p, 1), dtype=self.X.dtype, device=self.device)
            theta = torch.zeros(
                (n_interactions, 1), dtype=self.X.dtype, device=self.device
            )
            resid = -self.y
        else:
            resid = (
                self.X @ beta
                + self.prod_Z(self.X_, theta, self.meanZ, self.stdZ)
                - self.y
            )
        if Li is None or LX is None:
            LX = Lanczos(self.X, which="X", meanZ=None, stdZ=None, n_cv=20)
            Li = self.get_lipschitz_bb()
        zer = torch.zeros(1, dtype=self.X.dtype, device=self.device)
        tnew = 1
        k = 0
        if self.benchmark:
            _ = callback(k, beta, theta)
        else:
            _ = callback(k, beta, theta, resid=resid)
        # kkt0 = kkt_violation(self, beta, theta, zero=True,
        #                      bind="torch", resid=self.y)
        # eps *= kkt0
        # print("eps", eps, "kkt0", kkt0)
        kkt_ = kkt_violation(self, beta, theta, bind="torch", resid=-resid)
        # print(torch.linalg.norm(beta.view(-1), 0), torch.linalg.norm(theta.view(-1), 0))
        cond = kkt_ > eps and k < maxiter
        # print(kkt_)
        while cond:
            gbeta = 1 / n * self.XT @ resid
            nbeta = self.update_step(beta - n / LX * gbeta, n / LX, "beta")
            if self.use_acceleration:
                told = tnew
                tnew = (1 + sqrt(1 + 4 * told ** 2)) / 2
            diff_beta = nbeta - beta
            if torch.sum(torch.abs(diff_beta)) != 0:
                if self.use_acceleration:
                    nbeta += (told - 1) / tnew * diff_beta
                    resid = resid + self.X @ (nbeta - beta)
                else:
                    resid = resid + self.X @ diff_beta
            obegin = 0
            ntheta = theta.clone()
            for var in range(p):
                block_size = p - var
                next_ = obegin + block_size
                Zi = self.X_[:, var].view(-1, 1) * self.X_[:, var:]
                Zi -= self.meanZ[var]
                Zi /= self.stdZ[var]
                grad_theta_var = 1 / n * Zi.T @ resid
                updatable = ntheta[obegin:next_] - 1 / Li[var] * grad_theta_var
                ntheta[obegin:next_] = self.update_step(updatable, 1 / Li[var], "theta")
                diff_theta = ntheta[obegin:next_] - theta[obegin:next_]
                if not torch.isclose(torch.sum(torch.abs(diff_theta)), zer):
                    if self.use_acceleration:
                        ntheta[obegin:next_] += (told - 1) / tnew * diff_theta
                        resid += Zi @ (ntheta[obegin:next_] - theta[obegin:next_])
                    else:
                        resid += Zi @ diff_theta
                obegin = next_
            if self.benchmark:
                cond = callback(k, nbeta, ntheta)
            else:
                cond = callback(k, nbeta, ntheta, resid)
            beta, theta = nbeta, ntheta
            kkt_ = kkt_violation(self, beta, theta, bind="torch", resid=-resid)
            # print(k, kkt_)
            if kkt_ < eps and k > 10:
                break
            k += 1
            if recompute:
                if k % 100 == 0:
                    resid = (
                        self.X @ beta
                        + self.prod_Z(self.X_, theta, self.meanZ, self.stdZ)
                        - self.y
                    )

        self.beta = beta
        self.theta = theta
        if self.benchmark:
            callback(k - 1, self.beta, self.theta)
        else:
            callback(k - 1, self.beta, self.theta, resid=resid)
        self.cv = k
        return Li, LX, LZ

    def run_cbpg_full(self, maxiter, callback=None):
        """Run solver with q = p^2 interactions

        Args:
            maxiter (int): Maximum number of epochs
            callback (function, optional): criterion to stop the computation.
        """
        n, p = self.X.shape
        XT = self.X.T
        LX = torch.sqrt(Lanczos(self.X, which="X", meanZ=None, stdZ=None, n_cv=20)) / n
        Li = self.get_lipschitz_bb_full()
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
            gbeta = 1 / n * XT @ resid
            nbeta = self.update_step(beta - 1 / LX * gbeta, 1 / LX, "beta")
            diff_beta = nbeta - beta

            if torch.sum(torch.abs(diff_beta)) != 0:
                resid = resid + self.X @ diff_beta

            obegin = 0
            ntheta = theta.clone()
            block_size = p
            for var in range(p):
                next_ = obegin + block_size
                Zi = self.X_[:, var].view(-1, 1) * self.X_
                Zi -= self.meanZ[var]
                Zi /= self.stdZ[var]
                grad_theta_var = 1 / n * Zi.T @ resid
                updatable = ntheta[obegin:next_] - 1 / Li[var] * grad_theta_var
                ntheta[obegin:next_] = self.update_step_full(
                    updatable, 1 / Li[var], var
                )
                diff_theta = ntheta[obegin:next_] - theta[obegin:next_]
                if torch.abs(diff_theta).sum() != 0:
                    resid += Zi @ diff_theta
                obegin = next_
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
        self.cv = k

    def get_lipschitz_bb_full(self):
        """Get the list of the p majorations of Lipschitz constants"""
        n, p = self.X.shape
        ll = []
        LZ = torch.sqrt(
            Lanczos(self.X_, "Z", self.meanZ, self.stdZ, n_cv=20, full=True)
        )
        for var in range(p):
            Zi = self.X_[:, var].view(-1, 1) * self.X_
            Zi = (Zi - self.meanZ[var]) / self.stdZ[var]
            Li = torch.sqrt(Lanczos(Zi, "X", None, None, n_cv=20))
            ll.append(Li * LZ / n)
        return ll

    def get_lipschitz_bb(self):
        """Get the list of the p majorations of Lipschitz constants"""
        n, p = self.X.shape
        ll = []
        LZ = torch.sqrt(Lanczos(self.X_, "Z", self.meanZ, self.stdZ, n_cv=20))
        for var in range(p):
            Zi = self.X_[:, var].view(-1, 1) * self.X_[:, var:]
            Zi = (Zi - self.meanZ[var]) / self.stdZ[var]
            Li = torch.sqrt(Lanczos(Zi, "X", None, None, n_cv=20))
            ll.append(Li * LZ / n)
        return ll

    def update_step(self, x, lambda_, which="beta"):
        which = 0 if which == "beta" else 1
        val = lambda_ * self.alphas[which]
        thresh = x - torch.clamp(x, -val, val)
        return 1 / (1 + lambda_ * self.alphas[which + 2]) * thresh

    def update_step_full(self, x, lambda_, block):  # lambda_ = 1/LZ
        val = lambda_ * self.alphas[1]
        thresh_diag = 1 / (1 + lambda_ * self.alphas[3])
        thresh_out = 1 / (1 + 2 * lambda_ * self.alphas[3])
        thresh = x - torch.clamp(x, -val, val)

        thresh[:block] *= thresh_out
        thresh[block] *= thresh_diag
        thresh[(block + 1) :] *= thresh_out
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
            _, _, meanZ_sm, stdZ_sm = cpt_mean_std(self.X_, bind="torch")
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
            torch.tensor([-1e6], device=self.device, dtype=self.X.dtype),
            torch.linalg.norm(self.y, 2) ** 2,
            bind="torch",
        )
        primal, dual, gap, nu, cst_dual = res
        return primal, dual, gap, nu, cst_dual
