""""
=========================================================
Compare lispchitz cst upper bounds for different methods
=========================================================

We use the operator norm. We want to compare the time to
get the lipschitz constant and its value.
We do not use CUDA acceleration (only CPU).

Methods to compare: ||Z_{i:}.T @ Z|| <= ?, 1 <= i <= p
    - ||Z_{i:}|| ||Z||
    - sqrt(nlines * ncols) ||Z_{i:}.T @ Z||_max

Visu:
    - 2 plots
        Time: (1 horizontal line for each)
            - x = time for each method
            - y = method
        Values: (1 curve for each)
            - y = value of the lispchitz constant for each method
            - x = nb of the block (so p ticks)
"""

import torch
import time
import seaborn as sns
from interactionsmodel.data_management import download_data
from interactionsmodel import path_data
import numpy as np
import os
from interactionsmodel.utils import PRODS
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from interactionsmodel.utils import make_Z, cpt_mean_std, power_method, Lanczos
from interactionsmodel import path_tex
import warnings

sns.set()
use_cuda = torch.cuda.is_available()
dtype = torch.float32
device = "cuda" if use_cuda else "cpu"
print(f"Using {device}.")
MAXITER = int(1e6)
path_target = os.path.join(
    path_data, "Data", "genes_data_predicted_and_predictive_variables"
)


################################
# Generate the data
################################


def get_data(n=100, p=10, return_lip=False, seed=11235813):
    """Generate dataset and lipschitz constants for each
    block of the interaction matrix (without doubles features).

    Args:
        n (int, optional): Number of samples.
            Defaults to 100.
        p (int, optional): Number of features
            Defaults to 10.

    Returns:
        X (array, (n,p)): data generated standardized
        X_ (array, (n,p)): data generated
        meanZ (list): list of means by blocks of Z
        stdZ (list): list of std by blocks of Z
        lip (list): the p lispschitz constants of Z_{i:}.T @ Z., i <= p
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, _ = download_data(path_target, all_regulatory_regions=False)
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

    # X, _ = make_regression(n, p)

    X_ = X.copy()
    X = torch.from_numpy(X).float().to(device)
    X_ = torch.from_numpy(X_).float().to(device)
    meanX, stdX, meanZ, stdZ = cpt_mean_std(X, full=False, bind="torch")
    X -= meanX
    X /= stdX
    print("----- Finished computing means and std --------")
    n, p = X.shape
    if return_lip:
        Z = make_Z(X_, bind="torch")
        Z -= torch.mean(Z, axis=0)
        Z /= torch.std(Z, unbiased=False, axis=0)
        lip = []
        for var in range(p):
            Zi = X_[:, var].view(-1, 1) * X_[:, var:]
            Zi -= meanZ[var]
            Zi /= stdZ[var]
            Z_tmp = Zi.T @ Z
            lip.append(torch.linalg.norm(Z_tmp, 2))
        print("---- Finished computing exact Lipschitz constants ----")
        X, X_ = X.to(device), X_.to(device)
        _, _, meanZ, stdZ = cpt_mean_std(X_, full=False, bind="torch")
        return X, X_, meanZ, stdZ, lip
    else:
        X, X_ = X.to(device), X_.to(device)
        return X, X_, meanZ, stdZ, None


################################
# Methods to compare
################################


def maj_split(X, X_, meanZ, stdZ):
    n, p = X.shape
    ll = []
    LZ = torch.sqrt(power_method(X_, "Z", meanZ, stdZ, eps=1e-7, maxiter=100))
    for var in range(p):
        Zi = X_[:, var].view(-1, 1) * X_[:, var:]
        Zi = (Zi - meanZ[var]) / stdZ[var]
        Li = torch.linalg.norm(Zi, 2)
        ll.append(Li * LZ)
    return ll


def maj_max(X, X_, meanZ, stdZ):
    n, p = X.shape
    q = int(p * (p + 1) / 2)
    ll = []
    for varZi in range(p):  # for each lip cst
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
        ll.append(sqrt(p_tilde * q) * maxi)
    return ll


def maj_maj_max(X, X_, meanZ, stdZ):
    n, p = X.shape
    q = int(p * (p + 1) / 2)
    ll = [sqrt((p - var) * q) * torch.tensor(n) for var in range(p)]
    return ll


def pm(X, X_, meanZ, stdZ):
    n, p = X.shape
    prods = PRODS["small"]
    prod = prods[0]
    prodT = prods[1]
    size = int(p * (p + 1) / 2)
    ll = []
    device = X.device
    maxiter = 50
    for var in range(p):
        z = torch.randn((size, 1), dtype=X.dtype, device=device)
        z /= torch.linalg.norm(z)
        val = torch.tensor([0], dtype=X.dtype, device=device)
        Zi = X_[:, var].view(-1, 1) * X_[:, var:]
        Zi = (Zi - meanZ[var]) / stdZ[var]
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
            warnings.warn(
                "Warning ----------- Power iteration method did not converge !"
            )
        ll.append(torch.sqrt(val))
    return ll


def lanczos(X, X_, meanZ, stdZ):
    n, p = X.shape
    ll = []
    n_cv = 20
    Lz = torch.sqrt(Lanczos(X_, "Z", meanZ, stdZ))
    for var in range(p):
        n_cv = min(n_cv, p - var)
        Zi = X_[:, var].view(-1, 1) * X_[:, var:]
        Zi = (Zi - meanZ[var]) / stdZ[var]
        ll.append(torch.sqrt(Lanczos(Zi, "X", None, None, n_cv)) * Lz)
    return ll


#######################
# Running the benchmark
#######################


def run_routine(X, X_, meanZ, stdZ, routine):
    t_0 = time.perf_counter()
    lips = routine(X, X_, meanZ, stdZ)
    if use_cuda:
        torch.cuda.synchronize()
    time_ = time.perf_counter() - t_0
    return time_, lips


def benchmark(X, X_, meanZ, stdZ, dic_args, n_repeat=20):
    time_plot = []
    for n in range(n_repeat):
        time_, lip = run_routine(
            X.clone(), X_.clone(), meanZ, stdZ, dic_args["routine"]
        )
        time_plot.append(time_)
        print(f"Finished {n} out of {n_repeat}", f"time = {time_:.3f}s." "")
    time_plot = np.median(time_plot)
    return [time_plot], lip


def match_routine_args(routine_name):
    dict_args = {
        "consistency": {"routine": maj_split},
        "max": {"routine": maj_max},
        "max_CS": {"routine": maj_maj_max},
        "exact power_method": {"routine": pm},
        "lanczos": {"routine": lanczos},
    }
    return dict_args[routine_name]


def run_full_benchmark(routines_names, n=None, p=None, n_repeat=20):
    results = {name: {"times": [], "lipschitz": []} for name in routines_names}
    X, X_, meanZ, stdZ, lip = get_data(n=n, p=p, return_lip=False)
    for routine in routines_names:
        print(
            f"\033[1;%dm #########" % 34 + "Begin " + routine + "#########\033[0m"
        )  # noqa
        dict_args = match_routine_args(routine)
        times, lip_rout = benchmark(X, X_, meanZ, stdZ, dict_args, n_repeat=n_repeat)
        results[routine]["times"].extend(times)
        results[routine]["lipschitz"].extend(lip_rout)
    return results, lip


if __name__ == "__main__":

    routines_names = ["consistency", "max_CS", "exact power_method"]  # max
    plot_names = [r"$||Z_{\mathcal{B}_q(i)}|||Z||$", r"$n\sqrt{qp_i}$", "Power Method"]
    n, p = 100, 150

    results, real_lip = run_full_benchmark(routines_names, n, p, n_repeat=1)

    markers = ["+", "o", "^", "|", "*"]
    fig, ax = plt.subplots()
    y_pos = np.arange(len(routines_names))
    vals_times = [results[name]["times"][0] for name in routines_names]

    ax.barh(y_pos, vals_times, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(routines_names)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_xscale("log")
    fig.tight_layout()
    plt.show(block=False)

    x = np.arange(531)
    fig = plt.figure()
    if real_lip is not None:
        plt.plot(x, [real.cpu() for real in real_lip], label="exact_values")
    for idx, name in enumerate(routines_names):
        res = [
            results[name]["lipschitz"][i].cpu().item()
            for i in range(len(results[name]["lipschitz"]))
        ]
        name = plot_names[idx]
        plt.plot(x, res, marker=markers[idx], label=name)
    plt.xlabel(r"Block number $i$")
    plt.ylabel(r"Bounds on $||Z_{\mathcal{B}_q(i)}^\top Z||_2$")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            path_tex,
            "blocks_interactions",
            "prebuilt_images",
            "lipschitz_values_genomdata.pdf",
        )
    )
    plt.show()
