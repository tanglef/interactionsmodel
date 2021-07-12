"""
====================================
Paths with warmstart
====================================
"""
import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import interactionsmodel  # noqa for setting numba/numpy threads
from interactionsmodel.solvers import CD
from interactionsmodel.solvers import CBPG_CS as CBPG
from interactionsmodel import path_data, path_tex
from interactionsmodel.utils import make_regression, kkt_violation
from interactionsmodel.utils import get_lambda_max, cpt_mean_std
import time
from interactionsmodel.data_management import download_data
from sklearn.model_selection import train_test_split

plt.rcParams.update({"font.size": 16})
sns.set()
path_target = os.path.join(
    path_data, "Data", "genes_data_predicted_and_predictive_variables"
)
torch.set_printoptions(6)


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
MAX_ITER = int(500)


def run_CD(X_train, y_train, X_test, y_test, summary, lambdas, eps, warmstart, name):
    lambdas_ = [np.array(lambdas[i]).astype("float32") for i in range(len(lambdas))]
    cd = CD(X_train, y_train, lambdas_[0])  # init outside
    p = X.shape[1]
    n_inter = int(p * (p + 1) / 2)
    cd.run(
        1,
        eps=1.0,
        beta=np.zeros(p).astype("float32"),  # numba beta,theta !
        theta=np.zeros(n_inter).astype("float32"),
    )  # warmup
    ttotal = []
    for idx, pen in enumerate(lambdas_):
        begin = time.perf_counter()
        if warmstart and idx > 0:
            cd.run(MAX_ITER, eps=eps, lambdas=pen, beta=cd.beta, theta=cd.theta)
        else:
            cd.run(MAX_ITER, eps=eps, lambdas=pen)
        tend = time.perf_counter()
        ttotal.append(tend - begin)

        summary["nnb"].append(np.linalg.norm(cd.beta, 0).item())
        summary["nnt"].append(np.linalg.norm(cd.theta, 0).item())
        res = y_test.flatten() - (X_test - cd.meanX) / cd.stdX @ cd.beta
        n, p = X_test.shape
        theta_prod = np.zeros((n,))
        jj = 0
        for j1 in range(p):
            for j2 in range(j1, p):
                Z_tmp = X_test[:, j1] * X_test[:, j2]
                Z_tmp = (Z_tmp - cd.meanZ[jj]) / cd.stdZ[jj]
                theta_prod += Z_tmp * cd.theta[jj]
                jj += 1
        res -= theta_prod
        summary["mse"].append(np.linalg.norm(res, 2) ** 2 / len(res))
        print(
            "CV in:",
            cd.cv,
            ttotal[-1],
            "with kkt",
            kkt_violation(cd, cd.beta, cd.theta),
            "and gap",
            cd.get_dual_gap()[2],
        )

        summary["method"].append(name)
    print(
        f"\t finished CD [warmstart={warmstart}] with {cd.cv} epochs in {ttotal[-1]}."
    )  # noqa
    summary["times"].append(ttotal)


def run_CBPG(
    X_train,
    y_train,
    X_test,
    y_test,
    summary,
    lambdas,
    eps,
    use_acceleration,
    warmstart,
    recompute,
    name,
):
    lambdas_ = [
        torch.from_numpy(np.array(lambdas[i]).astype("float32")).float().to(device)
        for i in range(len(lambdas))
    ]
    cbpg = CBPG(
        torch.from_numpy(X_train).contiguous().to(device).float(),
        torch.from_numpy(y_train).view(-1, 1).to(device).float(),
        lambdas_[0],
        device=device,
        use_acceleration=use_acceleration,
    )
    cbpg.run(1, eps=1.0)
    Li, LX, LZ = None, None, None
    torch.cuda.synchronize()
    ttotal = []
    for idx, pen in enumerate(lambdas_):
        begin = time.perf_counter()
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
        torch.cuda.synchronize()
        tend = time.perf_counter()
        ttotal.append(tend - begin)

        summary["nnb"].append(
            np.linalg.norm(cbpg.beta.cpu().numpy().flatten(), 0).item()
        )
        summary["nnt"].append(
            np.linalg.norm(cbpg.theta.cpu().numpy().flatten(), 0).item()
        )

        X_test_ = torch.from_numpy(X_test).float().to(device) - cbpg.meanX
        X_test_ /= cbpg.stdX
        res = (
            torch.from_numpy(y_test).float().to(device).view(-1, 1)
            - X_test_ @ cbpg.beta
        )
        res -= cbpg.prod_Z(
            torch.from_numpy(X_test).float().to(device),
            cbpg.theta,
            cbpg.meanZ,
            cbpg.stdZ,
        )
        res = res.flatten()
        summary["mse"].append((torch.linalg.norm(res, 2) ** 2 / len(res)).cpu().numpy())
        summary["method"].append(name)
        print(
            "CV in:",
            cbpg.cv,
            ttotal[-1],
            "with kkt",
            kkt_violation(cbpg, cbpg.beta, cbpg.theta, bind="torch"),
            "and gap",
            cbpg.get_dual_gap()[2],
        )
    torch.cuda.synchronize()
    print(
        f"\t finished CBPG [acceleration={use_acceleration},warmstart={warmstart},recompute={recompute}] with {cbpg.cv} epochs  in {ttotal[-1]}."
    )  # noqa
    summary["times"].append(ttotal)


def run_all(X, y, lambdas):
    X = X.copy().astype("float32")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    y = y.astype("float32")
    summary = {
        "times": [],
        "ratio": [],
        "method": [],
        "eps": [],
        "nnb": [],
        "nnt": [],
        "mse": [],
    }
    for eps in [1e-4]:
        print(f"######## EPS = {eps} #######")
        run_CD(
            X_train,
            y_train,
            X_test,
            y_test,
            summary,
            lambdas,
            eps,
            True,
            name="CD+warmstart",
        )

        run_CBPG(
            X_train,
            y_train,
            X_test,
            y_test,
            summary,
            lambdas,
            eps,
            use_acceleration=False,
            warmstart=True,
            recompute=True,
            name="CBPG+warmstart",
        )

        # run_CBPG(
        #     X_train, y_train, X_test, y_test, summary, lambdas, eps,
        #     use_acceleration=True, warmstart=True, recompute=True,
        #     name="CBPG+acc+warmstart"
        # )
    return summary


if __name__ == "__main__":
    save_fig = True

    #############################
    # Load the data
    #############################
    global simulated
    simulated = True
    print(f"Cuda is {used_cuda}.")
    if simulated is True:
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
    elif simulated == "genom":
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
        X[:, :60]

    elif simulated == "MRI":
        data = pd.read_csv(
            os.path.join(path_target, "..", "MRI", "slice_localization_data.csv")
        )
        data_numpy = data.to_numpy()
        y = data_numpy[:, -1]
        X = data_numpy[:, 1:-1]
        _, sX, _, sZ = cpt_mean_std(X)
        X = X[:, sX != 0]
        X = np.delete(
            X,
            [28, 95, 96, 106, 250, 251, 258, 289, 290, 298, 299, 306, 323, 342],
            axis=1,
        )
        X, _ = np.unique(X, axis=1, return_index=True)
    n_samples, n_features = X.shape
    print(f"Dimensions are n={n_samples} and p={n_features}.")
    n_lambda = 10
    title = "path_e4_warm"
    l1_ratio = 0.9
    lambda_1, lambda_2 = get_lambda_max(X, y, bind="numpy", l1_ratio=l1_ratio)
    print(f"Lambda max are {lambda_1} and {lambda_2}")
    lambda_1 = max(lambda_1, lambda_2)

    # if simulated:
    grid_l = np.logspace(np.log10(lambda_1 / 1.1), np.log10(lambda_1 / 100), n_lambda)
    l_alpha = [
        (
            alpha * l1_ratio,
            alpha * l1_ratio,
            (1 - l1_ratio) * alpha,
            (1 - l1_ratio) * alpha,
        )
        for alpha in grid_l
    ]
    # l_alpha = l_alpha[:4]
    # grid = grid[:4]
    # else:
    #     grid = np.logspace(np.log10(lambda_1),
    #                        np.log10(lambda_1 / 100), n_lambda)
    #     l_alpha = [(alpha, alpha,
    #                 lambda_1*20, lambda_1*20)
    #                for alpha in grid
    #                ]
    print("All the regularizations tested")
    for regs in l_alpha:
        print(regs)
    print("~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~")

    # to remove later on
    # X = X.copy().astype("float32")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    # y = y.astype("float32")
    # lambdas_ = [torch.from_numpy(
    #         np.array(l_alpha[i]).astype("float32")
    #     ).float().to(device)
    #             for i in range(len(l_alpha))]
    # cbpg = CBPG(torch.from_numpy(X_train).contiguous().to(device).float(),
    #             torch.from_numpy(y_train).view(-1, 1).to(device).float(),
    #             lambdas_[0],
    #             device=device,
    #             use_acceleration=True,
    #             )
    # _ = cbpg.run(0)

    summary = run_all(X.astype("float32"), y.astype("float32"), l_alpha)
    print(summary)
    plt.figure()
    plt.plot(np.flip(grid_l), np.flip(summary["times"][0]), label="CD+warmstart")
    plt.plot(np.flip(grid_l), np.flip(summary["times"][1]), label="CBPG+warmstart")
    plt.legend()
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(path_save, "simu_ratio09.pdf"))
    plt.show()
