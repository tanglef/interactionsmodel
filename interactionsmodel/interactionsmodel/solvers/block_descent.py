"""
=======================
BLOCK GRADIENT DESCENT (LEGACY PRESENCE: NOT UPDATED!)
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
from interactionsmodel.utils import PRODS, power_method, cpt_mean_std, dual_gap_enet


class CBPG:
    def __init__(self, X, y, alphas, dtype, full=False):
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
        self.X_ = X.clone()
        self.X = X.clone()  # make sure we don't modify the original
        self.y = y.clone()
        self.alphas = alphas
        self.dtype = dtype
        prods = PRODS["full"] if full else PRODS["small"]
        self.prod_Z = prods[0]
        self.prod_ZT = prods[1]
        self.full = full
        self.meanX, self.stdX, self.meanZ, self.stdZ = cpt_mean_std(
            X, full=self.full, bind="torch"
        )
        self.X -= self.meanX
        self.X /= self.stdX

    def run(self, maxiter, eps=1e-7, maxiter_power=500, eps_power=1e-7, callback=None):
        if self.full:
            self.run_cbpg_full(
                maxiter, eps, maxiter_power, eps_power, callback=callback
            )
        else:
            self.run_cbpg(maxiter, eps, maxiter_power, eps_power, callback=callback)

    def run_cbpg(
        self, maxiter, eps=1e-7, maxiter_power=500, eps_power=1e-7, callback=None
    ):
        n, p = self.X.shape
        XT = self.X.T
        LX = torch.linalg.norm(self.X, 2) ** 2
        LZ = power_method(
            self.X_,
            which="Z",
            meanZ=self.meanZ,
            stdZ=self.stdZ,
            full=False,
            maxiter=maxiter_power,
            eps=eps_power,
        )
        Li = self.get_lipschitz_bb(LZ)
        n_interactions = int(p * (p + 1) / 2)
        beta = torch.zeros(p, 1).type(self.dtype)
        theta = torch.zeros(n_interactions, 1).type(self.dtype)
        r_old = -self.y
        for k in range(maxiter):
            if callback:
                callback(k, beta, theta)
            gbeta = 1 / n * XT @ r_old
            nbeta = self.update_step(beta - n / LX * gbeta, n / LX, "beta")
            r_new = r_old - self.X @ beta + self.X @ nbeta
            obegin = 0
            ntheta = theta.clone()
            for var in range(p):
                block_size = p - var
                next_ = obegin + block_size
                Zi = self.X_[:, var].view(-1, 1) * self.X_[:, var:]
                Zi -= self.meanZ[var]
                Zi /= self.stdZ[var]
                grad_theta_var = 1 / n * Zi.T @ r_new
                updatable = ntheta[obegin:next_] - n / Li[var] * grad_theta_var
                ntheta[obegin:next_] = self.update_step(
                    updatable,
                    n / Li[var],
                )
                r_new = r_new - Zi @ theta[obegin:next_] + Zi @ ntheta[obegin:next_]
                obegin = next_
            if torch.linalg.norm(r_new - r_old) < eps:
                break
            else:
                beta, theta = nbeta, ntheta
                r_old = r_new.clone()
        if callback:
            callback(maxiter, beta, theta)
        self.beta = beta
        self.theta = theta
        self.cv = k

    def run_cbpg_full(
        self, maxiter, eps=1e-7, maxiter_power=500, eps_power=1e-7, callback=None
    ):
        n, p = self.X.shape
        XT = self.X.T
        LX = torch.linalg.norm(self.X, 2) ** 2 / n
        LZ = (
            power_method(
                self.X_,
                which="Z",
                meanZ=self.meanZ,
                stdZ=self.stdZ,
                full=True,
                maxiter=maxiter_power,
                eps=eps_power,
            )
            / n
        )
        Li = self.get_lipschitz_bb_full(LX, LZ)
        n_interactions = int(p ** 2)
        beta = torch.zeros(p, 1).type(self.dtype)
        theta = torch.zeros(n_interactions, 1).type(self.dtype)
        r_old = -self.y
        for k in range(maxiter):
            if callback:
                callback(k, beta, theta)
            cv = k
            gbeta = 1 / n * XT @ r_old
            nbeta = self.update_step(beta - 1 / LX * gbeta, 1 / LX, "beta")
            r_new = r_old - self.X @ beta + self.X @ nbeta
            obegin = 0
            ntheta = theta.clone()
            block_size = p
            for var in range(p):
                next_ = obegin + block_size
                Zi = self.X_[:, var].view(-1, 1) * self.X_
                Zi -= self.meanZ[var]
                Zi /= self.stdZ[var]
                grad_theta_var = 1 / n * Zi.T @ r_new
                updatable = ntheta[obegin:next_] - 1 / Li[var] * grad_theta_var
                ntheta[obegin:next_] = self.update_step_full(
                    updatable, 1 / Li[var], var
                )
                r_new = r_new - Zi @ theta[obegin:next_] + Zi @ ntheta[obegin:next_]
                obegin = next_
            if torch.linalg.norm(r_new - r_old) < eps:
                break
            else:
                beta, theta = nbeta, ntheta
                r_old = r_new.clone()
        if callback:
            callback(maxiter, beta, theta)
        self.beta = beta
        self.theta = theta
        self.cv = cv

    def get_lipschitz_bb_full(self, LZ, LX):
        """Get the list of the p majorations of Lipschitz constants"""
        ll = []
        p = self.X.shape[1]
        for var in range(p):
            val = torch.max(self.X_[:, var])
            ll.append(val * LX * LZ * self.X_.shape[0])
        return ll

    def get_lipschitz_bb(self, LZ):
        """Get the list of the p majorations of Lipschitz constants"""
        ll = []
        p = self.X.shape[1]
        for var in range(p):
            Zi = self.X_[:, var].view(-1, 1) * self.X_[:, var:]
            Zi -= self.meanZ[var]
            Zi /= self.stdZ[var]
            val = torch.linalg.norm(Zi, 2) ** 2
            ll.append(val * LZ)
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
            theta_chg = torch.zeros((int(p * (p + 1) / 2), 1)).type(beta.type())
            obegin = 0
            for idx in range(p):
                p_tilde = p - idx
                theta_chg[obegin] = theta[idx, idx]
                extra_row = theta[idx, (idx + 1) :].view(-1, 1)
                theta_chg[(obegin + 1) : (obegin + p_tilde)] = 2 * extra_row
                obegin += p_tilde
            _, _, meanZ_sm, stdZ_sm = cpt_mean_std(self.X_, bind="torch")
            thetaprod = PRODS["small"][0](self.X_, theta_chg, meanZ_sm, stdZ_sm)
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
            theta_chg = torch.zeros((int(p * (p + 1) / 2), 1)).type(beta.type())
            obegin = 0
            for idx in range(p):
                p_tilde = p - idx
                theta_chg[obegin] = theta[idx, idx]
                extra_row = theta[idx, (idx + 1) :].view(-1, 1)
                theta_chg[(obegin + 1) : (obegin + p_tilde)] = 2 * extra_row
                obegin += p_tilde
            _, _, meanZ_sm, stdZ_sm = cpt_mean_std(self.X_, bind="torch")
            thetaprod = PRODS["small"][0](self.X_, theta_chg, meanZ_sm, stdZ_sm)
        else:
            theta_chg = theta
            meanZ_sm, stdZ_sm = self.meanZ, self.stdZ
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
            meanZ_sm,
            stdZ_sm,
            n,
            p,
            torch.tensor([-1e6]),
            torch.linalg.norm(self.y, 2) ** 2,
            bind="torch",
            full=False,
        )
        primal, dual, gap, nu, cst_dual = res
        return primal, dual, gap, nu, cst_dual
