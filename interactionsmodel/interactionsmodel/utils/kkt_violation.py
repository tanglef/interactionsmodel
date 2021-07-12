import torch
import numpy as np
from interactionsmodel.utils.numba_fcts.cd_enet_utils import soft_thresholding
from numba import njit

#################################
# When working with callbacks
#################################


# with W = [X, Z]
def kkt_violation(instance, beta, theta, bind="np", resid=None, zero=False):
    if bind != "torch":
        return kkt_violation_np(instance, beta, theta, resid)
    else:
        return kkt_violation_tch(instance, beta, theta, resid, zero)


def kkt_violation_tch(instance, beta, theta, resid=None, zero=False):
    n, p = instance.n, instance.p
    l1, l2, l3, l4 = instance.alphas
    if resid is None:
        theta_prod = instance.prod_Z(instance.X_, theta, instance.meanZ, instance.stdZ)
        resid = instance.y - instance.X @ beta - theta_prod
    oval = torch.tensor([-1e1], device=beta.device, dtype=beta.dtype)
    obegin = 0
    for var in range(1 + p):  # 1 for X and p for nb blocks in Z
        if var == p:
            intermed = instance.X.T @ resid
            if not zero:
                intermed -= n * l3 * beta
            thresh = intermed - torch.clamp(intermed, -n * l1, n * l1)
        else:
            next_ = obegin + p - var
            Zi = instance.X_[:, var].view(-1, 1) * instance.X_[:, var:]
            Zi -= instance.meanZ[var]
            Zi /= instance.stdZ[var]
            intermed = Zi.T @ resid
            if not zero:
                intermed -= n * l4 * theta[obegin:next_]
            thresh = intermed - torch.clamp(intermed, -n * l2, n * l2)
            obegin = next_
        val = 1 / n * torch.max(torch.abs(thresh))
        oval = torch.maximum(oval, val)
    return oval


def kkt_violation_np(instance, beta, theta, resid=None):
    n, p = instance.n, instance.p
    l1, l2, l3, l4 = instance.alphas
    if resid is None:
        theta_prod = np.zeros((n,)).astype(instance.beta.dtype)
        jj = 0
        for j1 in range(p):
            for j2 in range(j1, p):
                Z_tmp = instance.X_[:, j1] * instance.X_[:, j2]
                Z_tmp = (Z_tmp - instance.meanZ[jj]) / instance.stdZ[jj]
                theta_prod += Z_tmp * theta[jj]
                jj += 1

        resid = instance.y - instance.X @ beta - theta_prod
    oval = -np.inf
    jj = 0
    for j1 in range(p):
        intermedX = instance.X[:, j1].T @ resid - n * l3 * beta[j1]
        threshX = intermedX - np.clip(intermedX, -n * l1, n * l1)
        oval = np.maximum(oval, 1 / n * np.abs(threshX))
        for j2 in range(j1, p):
            Z_tmp = instance.X_[:, j1] * instance.X_[:, j2]
            Z_tmp = (Z_tmp - instance.meanZ[jj]) / instance.stdZ[jj]
            intermed = Z_tmp.T @ resid - n * l4 * theta[jj]
            thresh = intermed - np.clip(intermed, -n * l2, n * l2)
            oval = np.maximum(oval, 1 / n * np.abs(thresh))
            jj += 1
    return oval


#######################################
# For the solvers
#######################################


@njit
def kkt_nb(X, X_, y, meanZ, stdZ, beta, theta, resid, alphas):
    n, p = X.shape
    l1, l2, l3, l4 = alphas
    oval = -1e1
    jj = 0
    for j1 in range(p):
        intermedX = X[:, j1].T @ resid - n * l3 * beta[j1]
        threshX = soft_thresholding(intermedX, n * l1)
        oval = np.maximum(oval, 1 / n * np.abs(threshX))
        for j2 in range(j1, p):
            Z_tmp = X_[:, j1] * X_[:, j2]
            Z_tmp = (Z_tmp - meanZ[jj]) / stdZ[jj]
            intermed = Z_tmp.T @ resid - n * l4 * theta[jj]
            thresh = soft_thresholding(intermed, n * l2)
            oval = np.maximum(oval, 1 / n * np.abs(thresh))
            jj += 1
    return oval
