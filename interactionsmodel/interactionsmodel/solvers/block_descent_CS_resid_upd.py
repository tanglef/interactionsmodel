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
from interactionsmodel.utils import dual_gap_enet, Lanczos
from interactionsmodel.solvers import CBPG_CS


class CBPG_CS_mod(CBPG_CS):
    """Use a sampler on the blocks to be updated"""

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
        super(CBPG_CS_mod, self).__init__(
            X,
            y,
            alphas,
            device,
            full=False,
            use_acceleration=False,
            benchmark=True,
            **kwargs
        )

    def run(self, maxiter, callback=None, **kwargs):

        if callback is None:

            def callback(k, beta, theta):
                return k < maxiter

        if self.full:
            raise NotImplementedError
        else:
            self.run_cbpg(maxiter, callback=callback)

    ######################
    # With sampler

    def run_cbpg(self, maxiter, callback=None):
        n, p = self.n, self.p
        XT = self.X.T
        LX = Lanczos(self.X, which="X", meanZ=None, stdZ=None, n_cv=20)
        Li = self.get_lipschitz_bb()
        l_pm = make_pm(Li)
        beta = torch.zeros((p, 1), dtype=self.X.dtype, device=self.device)
        theta = [
            torch.zeros((p - var, 1), dtype=self.X.dtype, device=self.device)
            for var in range(p)
        ]
        resid = -self.y
        ntheta = [tt.clone() for tt in theta]
        tnew = torch.tensor([1.0], dtype=self.X.dtype, device=self.device)
        k = 0
        if self.benchmark:
            _ = callback(k, beta, theta)
        else:
            _ = callback(k, beta, theta, resid=resid)
        cond = True
        while cond:
            if self.use_acceleration:
                told = tnew.clone()
                tnew = (1 + torch.sqrt(1 + 4 * told ** 2)) / 2
            gbeta = 1 / n * XT @ resid
            nbeta = self.update_step(beta - n / LX * gbeta, n / LX, "beta")
            diff_beta = nbeta - beta
            if torch.abs(diff_beta).sum() != 0:
                if self.use_acceleration:
                    nbeta += (told - 1) / tnew * diff_beta
                    resid = resid + self.X @ (nbeta - beta)
                else:
                    resid = resid + self.X @ diff_beta

            for _ in range(p):  # make p updates
                which = sampler(l_pm)
                if self.use_acceleration:
                    self.update_theta(which, 1 / Li[which], resid, ntheta, told, tnew)
                else:
                    self.update_theta(which, 1 / Li[which], resid, ntheta, None, None)
            if self.benchmark:
                cond = callback(k, nbeta, ntheta)
            else:
                cond = callback(k, nbeta, ntheta, resid)
            beta, theta = nbeta, ntheta
            k += 1

        self.beta = beta
        self.theta = torch.cat(theta)
        if self.benchmark:
            callback(k - 1, self.beta, self.theta)
        else:
            callback(k - 1, self.beta, self.theta, resid=resid)
        self.cv = k

    def update_theta(self, var, lipinv, resid, ntheta, told, tnew):  # sampler
        n = self.n
        Zi = self.X_[:, var].view(-1, 1) * self.X_[:, var:]
        Zi -= self.meanZ[var]
        Zi /= self.stdZ[var]
        grad_theta_var = 1 / n * Zi.T @ resid
        updatable = ntheta[var] - lipinv * grad_theta_var
        old_ntheta = ntheta[var].clone()
        ntheta[var] = self.update_step(updatable, lipinv, "theta")
        diff_theta = ntheta[var] - old_ntheta
        if self.use_acceleration:
            ntheta[var] += (told - 1) / tnew * diff_theta
            resid += Zi @ (ntheta[var] - old_ntheta)
        else:
            resid += Zi @ diff_theta

    # def update_theta(self, var, lipinv, resid, obegin, next_,
    #                  ktheta, ntheta, theta, told, tnew, k):  # young-ish
    #     n = self.n
    #     Zi = (self.X_[:, var].view(-1, 1) * self.X_[:, var:])
    #     Zi -= self.meanZ[var]
    #     Zi /= self.stdZ[var]
    #     grad_theta_var = 1 / n * Zi.T @ resid
    #     t = k % 5
    #     L = 1 / lipinv
    #     l_ = 0.2
    #     step = 2 / ((L + l_) + (L - l_) * (math.cos((t - .5) * math.pi / 5)))
    #     updatable = ntheta[obegin:next_] - step * grad_theta_var
    #     ntheta[obegin:next_] = self.update_step(updatable,
    #                                             step,
    #                                             )
    #     diff_theta = ntheta[obegin:next_] - theta[obegin:next_]
    #     if self.use_acceleration:
    #         ntheta[obegin:next_] += (told - 1) / tnew * diff_theta
    #         resid += Zi @ (ntheta[obegin:next_] -
    #                        theta[obegin:next_])
    #     else:
    #         resid += Zi @ diff_theta

    # def update_theta(self, var, lipinv, resid, obegin, next_,
    #                  ktheta, ntheta, theta, told, tnew):  # permutations
    #     n = self.n
    #     Zi = (self.X_[:, var].view(-1, 1) * self.X_[:, var:])
    #     Zi -= self.meanZ[var]
    #     Zi /= self.stdZ[var]
    #     perm = torch.randperm(Zi.shape[1], dtype=int)
    #     invperm = torch.argsort(perm)
    #     Zi = Zi[:, perm]
    #     grad_theta_var = 1 / n * Zi.T @ resid
    #     ntheta[obegin:next_] = ntheta[obegin:next_][perm]
    #     updatable = ntheta[obegin:next_] - lipinv * grad_theta_var
    #     ntheta[obegin:next_] = self.update_step(updatable,
    #                                             lipinv,
    #                                             )
    #     diff_theta = ntheta[obegin:next_] - theta[obegin:next_][perm]
    #     if self.use_acceleration:
    #         ntheta[obegin:next_] += (told - 1) / tnew * diff_theta
    #         resid += Zi @ (ntheta[obegin:next_] -
    #                        theta[obegin:next_][perm])
    #     else:
    #         resid += Zi @ diff_theta
    #     ntheta[obegin:next_] = ntheta[obegin:next_][invperm]
    #     ktheta[obegin:next_] = ntheta[obegin:next_]

    # def update_theta(self, var, lipinv, resid, obegin, next_,
    #                  ktheta, ntheta, theta, told, tnew):  # smaller blocks
    #     n, p = self.n, self.p
    #     Zi = (self.X_[:, var].view(-1, 1) * self.X_[:, var:])
    #     Zi -= self.meanZ[var]
    #     Zi /= self.stdZ[var]
    #     if var < p / 2:
    #         breaks = torch.linspace(0, Zi.shape[1], 5, dtype=int)
    #     else:
    #         breaks = torch.linspace(0, Zi.shape[1], 2, dtype=int)
    #     for idx, b in enumerate(breaks[:-1]):
    #         nextb = breaks[idx+1]
    #         oben = obegin + nextb
    #         grad_theta_var = 1 / n * Zi[:, b:nextb].T @ resid
    #         updatable = ntheta[(obegin+b):oben] - lipinv * grad_theta_var
    #         ntheta[(obegin+b):oben] = self.update_step(updatable,
    #                                                    lipinv,
    #                                                    )
    #         ktheta[(obegin+b):oben] = ntheta[(obegin+b):oben]
    #         diff_theta = ntheta[(obegin+b):oben] - theta[(obegin+b):oben]
    #         if self.use_acceleration:
    #             ntheta[(obegin+b):oben] += (told - 1) / tnew * diff_theta
    #             resid += Zi[:, b:nextb] @ (ntheta[(obegin+b):oben] -
    #                                        theta[(obegin+b):oben])
    #         else:
    #             resid += Zi[:, b:nextb] @ diff_theta

    def run_cbpg_full(
        self, maxiter, eps=1e-7, maxiter_power=500, eps_power=1e-7, callback=None
    ):
        raise NotImplementedError

    def get_lipschitz_bb(self):
        """Get the list of the p majorations of Lipschitz constants"""
        n, p = self.n, self.p
        ll = []
        LZ = torch.sqrt(Lanczos(self.X_, "Z", self.meanZ, self.stdZ, n_cv=20))
        for var in range(p):
            Zi = self.X_[:, var].view(-1, 1) * self.X_[:, var:]
            Zi = (Zi - self.meanZ[var]) / self.stdZ[var]
            Li = torch.sqrt(Lanczos(Zi, "X", None, None, n_cv=20))
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
        n = self.n
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


def make_pm(ll):
    n = len(ll)
    m = (n - 1).bit_length()
    l_pm = [ll + [0.0] * ((1 << m) - n)]
    for k in range(1, m + 1):
        temp = [l_pm[k - 1][2 * i + 1] + l_pm[k - 1][2 * i] for i in range(1 << m - k)]
        l_pm.append(temp)
    return l_pm


def sampler(l_pm):
    m = len(l_pm)
    idx, val = 0, l_pm[-1][0]
    unif = torch.rand(m, device=val.device, dtype=val.dtype)
    for k in range(m - 1, 0, -1):
        temp1, temp2 = l_pm[k - 1][2 * idx], l_pm[k - 1][2 * idx + 1]
        if unif[k] < temp1 / val:
            idx, val = 2 * idx, temp1
        else:
            idx, val = 2 * idx + 1, temp2
    return idx
