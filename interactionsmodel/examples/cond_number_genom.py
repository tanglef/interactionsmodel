import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
from interactionsmodel import path_data, path_tex
from interactionsmodel.data_management import download_data
from interactionsmodel.utils import cpt_mean_std, make_regression, Lanczos

plt.rcParams.update({"font.size": 16})
sns.set()


def bblock(X, X_, meanZ=None, stdZ=None):
    n, p = X.shape
    cond_Z = []
    cond_X = torch.linalg.norm(X, 2) / torch.linalg.norm(X, -2)
    for j in range(p):
        Zt = X_[:, j].view(-1, 1) * X_[:, j:]
        Zt -= meanZ[j]
        Zt /= stdZ[j]
        cond_Z.append(
            (torch.linalg.norm(Zt, 2) / torch.linalg.norm(Zt, -2)).cpu().numpy()
        )
    return cond_X.cpu().numpy(), cond_Z


#########################
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
    simulated = False
    snr = 10
    #############################
    # Load the data
    #############################

    print(f"Cuda is {used_cuda}.")

    if not simulated:
        # get the data
        X, y = download_data(path_target, all_regulatory_regions=False)
        # delete DNA Shape
        X = X[:, :531]
        # rearange nucleotide and dinucleotide
        X[:, :20] = X[
            :, np.hstack([np.array([4, 5, 6, 19, 0, 1, 2, 3]), np.arange(7, 19)])
        ]
        X[:, 20:40] = X[
            :, np.hstack([np.array([4, 5, 6, 19, 0, 1, 2, 3]), np.arange(7, 19)]) + 20
        ]
        X[:, 40:60] = X[
            :, np.hstack([np.array([4, 5, 6, 19, 0, 1, 2, 3]), np.arange(7, 19)]) + 40
        ]
        if fast_data:
            X = X[:, :10]
        X = torch.from_numpy(X).float().to(device)
    else:
        X, y, beta0, beta, theta = make_regression(
            20000,
            500,
            False,
            mu=10,
            sigma=0.5,
            seed=112358,
            snr=snr,
            sparse_inter=0.01,
            sparse_main=0.01,
            ltheta=-1e3,
            utheta=1e3,
            lbeta=-1e2,
            ubeta=1e2,
        )
        X = torch.from_numpy(X).float().to(device)
    X_ = X.clone()
    meanX, stdX, meanZ, stdZ = cpt_mean_std(X, full=False, bind="torch")
    X -= meanX
    X /= stdX
    print("~~~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~~~")

    cond_X, cond_Z = bblock(X, X_, meanZ, stdZ)
    print("Torch", cond_X)

    plt.figure()
    plt.plot(np.arange(0, len(cond_Z)), cond_Z)
    plt.xlabel("block")
    plt.ylabel("condition number")
    plt.yscale("log")
    if save_fig:
        plt.savefig(os.path.join(path_save, f"condition_number_genom.pdf"))  # noqa
    plt.show()

    # print(f"Lanczos X: {torch.sqrt(Lanczos(X, 'X'))}, Torch X: {torch.linalg.norm(X, 2)}")
    # # Approximations
    # ll_lanczos = []
    # ll_torch = []
    # p = X.shape[1]
    # for var in range(p):
    #     Zi = X_[:, var].view(-1, 1) * X_[:, var:]
    #     Zi = (Zi - meanZ[var]) / stdZ[var]
    #     Li = torch.sqrt(Lanczos(Zi, "X", None, None, n_cv=20))
    #     ll_lanczos.append(Li.cpu().numpy())
    #     ll_torch.append(torch.linalg.norm(Zi, 2).cpu().numpy())

    # plt.figure()
    # plt.plot(np.arange(0, len(ll_lanczos)), ll_lanczos, label="lanczos")
    # plt.plot(np.arange(0, len(ll_torch)), ll_torch, label="torch")
    # plt.xlabel("block")
    # plt.ylabel("Lambda max")
    # plt.yscale("log")
    # if save_fig:
    #     plt.savefig(os.path.join(path_save,
    #                              f"Maximum_eigen_value.pdf"))  # noqa
    plt.show()
