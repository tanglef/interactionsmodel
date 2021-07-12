"""
====================================
Barplots of running a path
====================================
"""

import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
import interactionsmodel  # noqa for setting numba/numpy threads
from interactionsmodel.solvers import CD
from interactionsmodel.solvers import CBPG_CS as CBPG
from interactionsmodel import path_data, path_tex
from interactionsmodel.utils import make_regression, kkt_violation
from interactionsmodel.utils import get_lambda_max
import time
from interactionsmodel.data_management import download_data
from sklearn.model_selection import train_test_split

plt.rcParams.update({"font.size": 16})
sns.set()
path_target = os.path.join(
    path_data, "Data", "genes_data_predicted_and_predictive_variables"
)


##########################
# Load and prepare data
###########################

# path to the data
path_save = os.path.join(path_tex, "blocks_interactions", "prebuilt_images")

# use CUDA if available
use_cuda = torch.cuda.is_available()
dtype = torch.float32
device = "cuda" if use_cuda else "cpu"
used_cuda = "used" if use_cuda else "not used"
EPS = 1e-3
MAX_ITER = int(5000)


def run_CD(X, y, lambdas, eps, warmstart):
    early_stop = False
    lambdas_ = [np.array(lambdas[i]).astype("float32") for i in range(len(lambdas))]
    cd = CD(X, y, lambdas_[0])  # init outside
    p = X.shape[1]
    n_inter = int(p * (p + 1) / 2)
    cd.run(
        1,
        eps=1.0,
        beta=np.zeros(p).astype("float32"),  # numba beta,theta !
        theta=np.zeros(n_inter).astype("float32"),
    )  # warmup
    begin = time.perf_counter()
    for idx, pen in enumerate(lambdas_):
        if warmstart and idx > 0:
            cd.run(MAX_ITER, eps=eps, lambdas=pen, beta=cd.beta, theta=cd.theta)
        else:
            cd.run(MAX_ITER, eps=eps, lambdas=pen)
        if cd.cv == MAX_ITER:
            early_stop = True
            print(f"CD did not converge for eps={eps}.")
            break
    total = time.perf_counter() - begin
    print(
        f"\t finished CD [warmstart={warmstart}] in {total:.3f}s with {cd.cv} epochs."
    )  # noqa

    return total, cd, early_stop


def run_CBPG(X, y, lambdas, eps, use_acceleration, warmstart, recompute):
    early_stop = False
    lambdas_ = [
        torch.from_numpy(np.array(lambdas[i]).astype("float32")).float().to(device)
        for i in range(len(lambdas))
    ]
    cbpg = CBPG(
        torch.from_numpy(X).contiguous().to(device).float(),
        torch.from_numpy(y).view(-1, 1).to(device).float(),
        lambdas_[0],
        device=device,
        use_acceleration=use_acceleration,
    )
    cbpg.run(1, eps=1.0)
    Li, LX, LZ = None, None, None
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for idx, pen in enumerate(lambdas_):
        if warmstart and idx > 0:
            Li, LX, LZ = cbpg.run(
                MAX_ITER,
                eps=eps,
                lambdas=pen,
                Li=Li,
                LX=LX,
                LZ=LZ,
                beta=cbpg.beta,
                theta=cbpg.theta,
                recompute=recompute,
                alphas=pen,
            )
        else:
            recompute_tmp = True if idx == 0 else recompute
            Li, LX, LZ = cbpg.run(
                MAX_ITER,
                eps=eps,
                lambdas=pen,
                Li=Li,
                LX=LX,
                LZ=LZ,
                recompute=recompute_tmp,
                alphas=pen,
            )
        if cbpg.cv == MAX_ITER:
            early_stop = True
            print(
                f"CBPG did not converge for eps={eps} [acceleration={use_acceleration},warmstart={warmstart},recompute={recompute}]."
            )  # noqa
            break
    torch.cuda.synchronize()
    total = time.perf_counter() - begin
    print(
        f"\t finished CBPG [acceleration={use_acceleration},warmstart={warmstart},recompute={recompute}] in {total:.3f}s with {cbpg.cv} epochs."
    )  # noqa
    return total, cbpg, early_stop


def update_summary(summary, total, instance, eps, name, idx, idx_ref):
    summary["time"].append(total)
    summary["method"].append(name)
    summary["eps"].append(str(eps))
    if name[0:2] == "CD":
        summary["nnb"].append(np.linalg.norm(instance.beta, 0).item())
        summary["nnt"].append(np.linalg.norm(instance.theta, 0).item())
        if len(name) == 2:
            idx_ref = idx
    else:
        summary["nnb"].append(
            np.linalg.norm(instance.beta.cpu().numpy().flatten(), 0).item()
        )
        summary["nnt"].append(
            np.linalg.norm(instance.theta.cpu().numpy().flatten(), 0).item()
        )
    summary["ratio"].append(summary["time"][idx] / summary["time"][idx_ref])
    idx += 1
    return idx, idx_ref


def run_all(X, y, lambdas):
    X = X.copy().astype("float32")
    y = y.astype("float32")
    summary = {"time": [], "ratio": [], "method": [], "eps": [], "nnb": [], "nnt": []}
    idx, idx_ref = 0, 0
    for eps in [1e-2, 1e-3]:
        print(f"######## EPS = {eps} #######")
        total, cd, early_stop_cd = run_CD(X, y, lambdas, eps, False)
        idx, idx_ref = update_summary(summary, total, cd, eps, "CD", idx, idx_ref)
        # assert not early_stop_cd

        total, cd, early_stop_cd = run_CD(X, y, lambdas, eps, True)
        idx, idx_ref = update_summary(
            summary, total, cd, eps, "CD+warmstart", idx, idx_ref
        )
        # assert not early_stop_cd

        total, cbpg, early_stop = run_CBPG(
            X, y, lambdas, eps, use_acceleration=False, warmstart=False, recompute=True
        )
        idx, idx_ref = update_summary(summary, total, cbpg, eps, "CBPG", idx, idx_ref)
        # assert not early_stop

        total, cbpg, early_stop = run_CBPG(
            X, y, lambdas, eps, use_acceleration=True, warmstart=False, recompute=True
        )
        idx, idx_ref = update_summary(
            summary, total, cbpg, eps, "CBPG+acc", idx, idx_ref
        )
        # assert not early_stop

        total, cbpg, early_stop = run_CBPG(
            X, y, lambdas, eps, use_acceleration=True, warmstart=True, recompute=False
        )
        idx, idx_ref = update_summary(
            summary, total, cbpg, eps, "CBPG+acc+warmstart", idx, idx_ref
        )
        # assert not early_stop

        total, cbpg, early_stop = run_CBPG(
            X, y, lambdas, eps, use_acceleration=True, warmstart=True, recompute=True
        )
        idx, idx_ref = update_summary(
            summary,
            total,
            cbpg,
            eps,
            "CBPG+acc+warmstart+recompte for all",
            idx,
            idx_ref,
        )
        # assert not early_stop

    return summary


if __name__ == "__main__":
    save_fig = True

    #############################
    # Load the data
    #############################
    global simulated
    simulated = False
    print(f"Cuda is {used_cuda}.")
    if simulated:
        snr = 10
        global beta, theta
        n_samples, n_features = 20000, 500
        n_inter = int(n_features * (n_features + 1) / 2)
        X, y, beta0, beta, theta = make_regression(
            n_samples,
            n_features,
            False,
            mu=0,
            sigma=1,
            seed=112358,
            snr=snr,
            sparse_inter=0.01,
            sparse_main=0.01,
            ltheta=-1e2,
            utheta=1e2,
            lbeta=-1e3,
            ubeta=1e3,
        )
    else:
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
        y = y[:, 0]
        n_samples, n_features = X.shape
    n_lambda = 10
    title = "barplot_genom_factor20"
    l1_ratio = 1
    lambda_1, lambda_2 = get_lambda_max(X, y, bind="numpy", l1_ratio=l1_ratio)
    print(f"Lambda max are {lambda_1} and {lambda_2}")
    lambda_1 = max(lambda_1, lambda_2)

    if simulated:
        if simulated:
            grid = np.logspace(
                np.log10(lambda_1 / 1.2), np.log10(lambda_1 / 100), n_lambda
            )
            l_alpha = [
                (
                    alpha * l1_ratio,
                    alpha * l1_ratio,
                    alpha * (1 - l1_ratio),
                    alpha * (1 - l1_ratio),
                )
                for alpha in grid
            ]
    else:
        grid = np.logspace(np.log10(lambda_1), np.log10(lambda_1 / 100), n_lambda)
        l_alpha = [(alpha, alpha, lambda_1 * 20, lambda_1 * 20) for alpha in grid]
    print("All the regularizations tested")
    for regs in l_alpha:
        print(regs)
    print("~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~")
    summary = run_all(X, y, l_alpha)
    import pandas as pd

    g = sns.catplot(
        data=pd.DataFrame(summary), kind="bar", x="eps", y="ratio", hue="method"
    )
    g.despine(left=True)
    g.set_axis_labels("precision", "time / time CD")
    g.legend.set_title("")
    if save_fig:
        plt.savefig(os.path.join(path_save, title + ".pdf"))  # noqa
    plt.show()
