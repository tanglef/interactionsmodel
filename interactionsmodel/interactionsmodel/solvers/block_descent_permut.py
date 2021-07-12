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
from interactionsmodel.utils import PRODS, cpt_mean_std, dual_gap_enet
from interactionsmodel.utils import power_method


class CBPG_permut:
    def __init__(self, X, y, alphas, device, full=False, use_acceleration=False):
        """
        Args:
            X (tensor): data of size (n,p)
            y (tensor): response column tensor of size (p,1)
            alphas (list): list of the four penalties of the problem in
                the following order: [l1_beta, l1_theta, l2_beta, l2_theta]
            dtype (torch type): mostly torch.cuda.FloatTensor or w/o cuda
            full (bool, optional): Use all the interactions or remove doubles.
                Defaults to False.
        """
        self.X_ = X.clone().to(device)
        self.X = X.clone().to(device)  # we don't modify the original
        self.y = y.clone().to(device)
        self.alphas = alphas
        self.n, self.p = X.shape
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
        if self.full:  # accelerate get_objective in the benchmarks
            _, _, self.meanZ_sm, self.stdZ_sm = cpt_mean_std(self.X_, bind="torch")
        else:
            self.meanZ_sm, self.stdZ_sm = self.meanZ, self.stdZ

    def run(self, maxiter, eps=1e-7, maxiter_power=500, eps_power=1e-7, callback=None):

        if callback is None:

            def callback(k, beta, theta):
                return k < maxiter

        if self.full:
            raise NotImplementedError
        else:
            self.maxiter_pm = maxiter_power
            self.eps_pm = eps_power
            self.run_cbpg(maxiter, eps, maxiter_power, eps_power, callback=callback)

    # def run_cbpg(self, maxiter, eps=1e-7,
    #              maxiter_power=500, eps_power=1e-7,
    #              callback=None):
    #     n, p = self.n, self.p
    #     XT = self.X.T
    #     LX = torch.linalg.norm(self.X, 2) ** 2
    #     Li = self.get_lipschitz_bb()
    #     n_interactions = int(p * (p + 1) / 2)
    #     beta = torch.zeros((p, 1), dtype=self.X.dtype, device=self.device)
    #     theta = torch.zeros((n_interactions, 1), dtype=self.X.dtype,
    #                         device=self.device)
    #     kbeta, ktheta = beta, theta
    #     resid = -self.y
    #     tnew = torch.tensor([1.], dtype=self.X.dtype, device=self.device)
    #     for k in range(maxiter):
    #         if callback:
    #             callback(k, kbeta, ktheta)
    #         if self.use_acceleration:
    #             told = tnew.clone()
    #             tnew = (1 + torch.sqrt(1 + 4 * told ** 2)) / 2
    #         gbeta = 1 / n * XT @ resid
    #         nbeta = self.update_step(beta - n / LX * gbeta,
    #                                  n / LX, "beta")
    #         kbeta = nbeta.clone()
    #         diff_beta = nbeta - beta
    #         if (diff_beta != 0).any():
    #             if self.use_acceleration:
    #                 nbeta += (told - 1) / tnew * diff_beta
    #                 resid = resid + self.X @ (nbeta - beta)
    #             else:
    #                 resid = resid + self.X @ diff_beta

    #         obegin = 0
    #         ntheta = theta.clone()
    #         ktheta = ntheta.clone()
    #         for var in range(p):
    #             block_size = p - var
    #             next_ = obegin + block_size
    #             if self.use_acceleration:
    #                 self.update_theta(var, n / Li[var], resid, obegin, next_,
    #                                   ktheta, ntheta, theta, told, tnew, k)
    #             else:
    #                 self.update_theta(var, n / Li[var], resid, obegin, next_,
    #                                   ktheta, ntheta, theta, None, None, k)
    #             obegin = next_
    #         if torch.linalg.norm(diff_beta) < eps and \
    #            torch.linalg.norm(ktheta - theta) < eps:
    #             beta, theta = kbeta, ktheta
    #             break
    #         else:
    #             beta, theta = nbeta, ntheta
    #             resid = resid.clone()
    #     if self.use_acceleration:
    #         self.beta = kbeta
    #         self.theta = ktheta
    #     else:
    #         self.beta = beta
    #         self.theta = theta
    #     if callback:
    #         callback(maxiter, self.beta, self.theta)
    #     self.cv = k

    ######################
    # With permutations

    def run_cbpg(
        self, maxiter, eps=1e-7, maxiter_power=500, eps_power=1e-7, callback=None
    ):
        n, p = self.n, self.p
        q = int(p * (p + 1) / 2)
        XT = self.X.T
        LX = torch.linalg.norm(self.X, 2) ** 2
        Li = self.get_lipschitz_bb()
        beta = torch.zeros((p, 1), dtype=self.X.dtype, device=self.device)
        theta = torch.zeros((q, 1), dtype=self.X.dtype, device=self.device)
        kbeta, ktheta = beta, theta
        resid = -self.y
        ntheta = theta.clone()
        tnew = torch.tensor([1.0], dtype=self.X.dtype, device=self.device)
        k = 0
        cond = True
        _ = callback(k, beta, theta)
        while cond:
            if self.use_acceleration:
                told = tnew.clone()
                tnew = (1 + torch.sqrt(1 + 4 * told ** 2)) / 2
            gbeta = 1 / n * XT @ resid
            nbeta = self.update_step(beta - n / LX * gbeta, n / LX, "beta")
            kbeta = nbeta.clone()
            diff_beta = nbeta - beta
            if (diff_beta != 0).any():
                if self.use_acceleration:
                    nbeta += (told - 1) / tnew * diff_beta
                    resid = resid + self.X @ (nbeta - beta)
                else:
                    resid = resid + self.X @ diff_beta
            obegin = 0
            ntheta = theta.clone()
            ktheta = ntheta.clone()
            for var in range(p):
                next_ = obegin + p - var
                if self.use_acceleration:
                    self.update_theta(
                        var,
                        1 / Li[var],
                        resid,
                        obegin,
                        next_,
                        ktheta,
                        ntheta,
                        theta,
                        told,
                        tnew,
                    )
                else:
                    self.update_theta(
                        var,
                        1 / Li[var],
                        resid,
                        obegin,
                        next_,
                        ktheta,
                        ntheta,
                        theta,
                        None,
                        None,
                    )
                obegin = next_
            cond = (
                callback(k, kbeta, ktheta)
                if self.use_acceleration
                else callback(k, beta, theta)
            )
            beta, theta = nbeta, ntheta
            k += 1
        if self.use_acceleration:
            self.beta = kbeta
            self.theta = ktheta
        else:
            self.beta = beta
            self.theta = theta
        self.cv = k

    def update_theta(
        self, var, lipinv, resid, obegin, next_, ktheta, ntheta, theta, told, tnew
    ):  # permutations
        n = self.n
        Zi = self.X_[:, var].view(-1, 1) * self.X_[:, var:]
        Zi -= self.meanZ[var]
        Zi /= self.stdZ[var]
        perm = torch.randperm(Zi.shape[1], dtype=int)
        invperm = torch.argsort(perm)
        Zi = Zi[:, perm]
        grad_theta_var = 1 / n * Zi.T @ resid
        ntheta[obegin:next_] = ntheta[obegin:next_][perm]
        updatable = ntheta[obegin:next_] - lipinv * grad_theta_var
        ntheta[obegin:next_] = self.update_step(
            updatable,
            lipinv,
        )
        diff_theta = ntheta[obegin:next_] - theta[obegin:next_][perm]
        if self.use_acceleration:
            ntheta[obegin:next_] += (told - 1) / tnew * diff_theta
            resid += Zi @ (ntheta[obegin:next_] - theta[obegin:next_][perm])
        else:
            resid += Zi @ diff_theta
        ntheta[obegin:next_] = ntheta[obegin:next_][invperm]
        if self.use_acceleration:
            ktheta[obegin:next_] = ntheta[obegin:next_]

    def run_cbpg_full(
        self, maxiter, eps=1e-7, maxiter_power=500, eps_power=1e-7, callback=None
    ):
        raise NotImplementedError

    def get_lipschitz_bb(self):
        """Get the list of the p majorations of Lipschitz constants"""
        n, p = self.n, self.p
        ll = []
        LZ = torch.sqrt(
            power_method(
                self.X_,
                "Z",
                self.meanZ,
                self.stdZ,
                eps=self.eps_pm,
                maxiter=self.maxiter_pm,
            )
        )
        for var in range(p):
            Zi = self.X_[:, var].view(-1, 1) * self.X_[:, var:]
            Zi = (Zi - self.meanZ[var]) / self.stdZ[var]
            Li = torch.linalg.norm(Zi, 2)
            ll.append(Li * LZ / n)
        return ll

    def get_lipschitz_bb_full(self):
        """Get the list of the p majorations of Lipschitz constants"""
        raise NotImplementedError

    def update_step(self, x, lambda_, which="beta"):
        which = 0 if which == "beta" else 1
        val = lambda_ * self.alphas[which]
        thresh = x - torch.clamp(x, -val, val)
        return 1 / (1 + lambda_ * self.alphas[which + 2]) * thresh

    def update_step_full(self, x, lambda_, block):
        raise NotImplementedError

    def get_objective(self, beta=None, theta=None):
        """Compute the objective value"""
        if beta is None or theta is None:
            beta, theta = self.beta, self.theta
        n, p = self.n, self.p
        if self.full:
            raise NotImplementedError
        else:
            thetaprod = self.prod_Z(self.X_, theta, self.meanZ, self.stdZ)
        mc_int = self.y - self.X @ beta - thetaprod
        mc_int = mc_int.flatten()
        mc = 0.5 / n * (mc_int @ mc_int)
        beta_l1 = self.alphas[0] * torch.abs(beta).sum()
        beta_l2 = self.alphas[2] / 2 * (beta ** 2).sum()
        theta_l1 = self.alphas[1] * torch.abs(theta).sum()
        theta_l2 = self.alphas[3] / 2 * torch.linalg.norm(theta, 2) ** 2
        return mc + beta_l1 + beta_l2 + theta_l1 + theta_l2

    def get_dual_gap(self, beta=None, theta=None):
        if beta is None and theta is None:
            beta, theta = self.beta, self.theta
        n, p = self.n, self.p
        if self.full:
            raise NotImplementedError
        else:
            thetaprod = self.prod_Z(self.X_, theta, self.meanZ, self.stdZ)
        resid = self.y - self.X @ beta - thetaprod
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
