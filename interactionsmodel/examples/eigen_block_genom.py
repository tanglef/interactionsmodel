import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
from interactionsmodel import path_data, path_tex
from interactionsmodel.data_management import download_data
from interactionsmodel.utils import power_method, cpt_mean_std, PRODS, make_Z
from math import sqrt

plt.rcParams.update({"font.size": 16})
sns.set()


def maj_split(X, X_, meanZ, stdZ, var):
    n, p = X.shape
    LZ = torch.sqrt(power_method(X_, "Z", meanZ, stdZ, eps=1e-7, maxiter=int(1e6)))
    Zi = X_[:, var].view(-1, 1) * X_[:, var:]
    Zi = (Zi - meanZ[var]) / stdZ[var]
    Li = torch.linalg.norm(Zi, 2)
    return Li, LZ, Li * LZ


def pm(X, X_, meanZ, stdZ, var):
    n, p = X.shape
    prods = PRODS["small"]
    prod = prods[0]
    prodT = prods[1]
    size = int(p * (p + 1) / 2)
    device = X.device
    z = torch.randn((size, 1), dtype=X.dtype, device=device)
    z /= torch.linalg.norm(z)
    val = torch.tensor([0], dtype=X.dtype, device=device)
    Zi = X_[:, var].view(-1, 1) * X_[:, var:]
    Zi = (Zi - meanZ[var]) / stdZ[var]
    maxiter = 100 * Zi.shape[1]
    for k in range(maxiter):
        z1 = prod(X_, z, meanZ, stdZ)
        z2 = Zi.T @ z1
        z3 = Zi @ z2
        z_new = prodT(X_, z3, meanZ, stdZ)
        # Rayleigh quotient
        val_new = val
        val = z_new.view(-1) @ z.view(-1)
        if ((val - val_new) ** 2).sum() <= 1e-4:
            break
        z = z_new / torch.linalg.norm(z_new)
    if k + 1 == maxiter:
        print("Warning ----------- Power iteration method did not converge !")
    return torch.sqrt(val)


def maj_max(X, X_, meanZ, stdZ, varZi):
    n, p = X.shape
    q = int(p * (p + 1) / 2)
    p_tilde = p - varZi
    maxi = torch.tensor(0).type(dtype)
    Zi = X_[:, varZi].view(-1, 1) * X_[:, varZi:]
    Zi = (Zi - meanZ[varZi]) / stdZ[varZi]
    Zi = Zi.T
    for varZ in range(p):  # Zi.T @ Z by block
        Zj = X_[:, varZ].view(-1, 1) * X_[:, varZ:]
        Zj = (Zj - meanZ[varZ]) / stdZ[varZ]
        Z_tmp = Zi @ Zj
        maxi = torch.max(maxi, torch.max(torch.abs(Z_tmp)))
    return sqrt(p_tilde * q) * maxi


def maj_maj_max(X, X_, meanZ, stdZ, var):
    n, p = X.shape
    q = int(p * (p + 1) / 2)
    ll = sqrt((p - var) * q) * torch.tensor(n)
    return ll


##########################
# Load and prepare data
###########################

# path to the data
path_target = os.path.join(
    path_data, "Data", "genes_data_predicted_and_predictive_variables"
)
path_save = os.path.join(path_tex, "blocks_interactions", "prebuilt_images")

# use CUDA if available
use_cuda = torch.cuda.is_available()
dtype = torch.float32
device = "cuda" if use_cuda else "cpu"
used_cuda = "used" if use_cuda else "not used"
EPS = 1e-7
MAXITER = int(1e3)
EPS_PM = 1e-7


if __name__ == "__main__":
    save_fig = False
    fast_data = False  # take only 10 features
    #############################
    # Load the data
    #############################

    print(f"Cuda is {used_cuda}.")

    # get the data
    X, y = download_data(path_target, all_regulatory_regions=False)
    # delete DNA Shape
    X = X[:, :531]
    # rearange nucleotide and dinucleotide
    X[:, :20] = X[:, np.hstack([np.array([4, 5, 6, 19, 0, 1, 2, 3]), np.arange(7, 19)])]
    X[:, 20:40] = X[
        :, np.hstack([np.array([4, 5, 6, 19, 0, 1, 2, 3]), np.arange(7, 19)]) + 20
    ]
    X[:, 40:60] = X[
        :, np.hstack([np.array([4, 5, 6, 19, 0, 1, 2, 3]), np.arange(7, 19)]) + 40
    ]
    if fast_data:
        X = X[:, :10]
    X = torch.from_numpy(X).float().to(device)
    X_ = X.clone()
    meanX, stdX, meanZ, stdZ = cpt_mean_std(X, full=False, bind="torch")
    X -= meanX
    X /= stdX
    print("~~~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~~~")

    block = 5
    print(f"###### {block}-th block")
    cst = pm(X, X_, meanZ, stdZ, block)
    print("end power")
    Li, LZ, cst_majsplot = maj_split(X, X_, meanZ, stdZ, block)
    print("end consistency")
    cst_cs = maj_maj_max(X, X_, meanZ, stdZ, block)
    cst_maj_max = maj_max(X, X_, meanZ, stdZ, block)
    print(f"Real = {cst}, maj = {cst_majsplot}, cs = {cst_cs}, max = {cst_maj_max}")
    print(cst < cst_cs)
    print(cst < cst_majsplot)
    print(cst < cst_maj_max)

    # print("\n\n With linalg")
    # Z = make_Z(X_)
    # Z -= torch.mean(Z, axis=0)
    # Z /= torch.std(Z, unbiased=False, axis=0)
    # Zi = X_[:, block].view(-1, 1) * X_[:, block:]
    # Zi -= meanZ[block]
    # Zi /= stdZ[block]
    # print(f"Linalg = {torch.linalg.norm(Zi.T @ Z, 2)}")
    # normZ = torch.linalg.norm(Z, 2)
    # normZi = torch.linalg.norm(Zi, 2)
    # print(normZ, LZ)
    # print(normZi, Li)
    # print(normZ*normZi, cst_majsplot)
