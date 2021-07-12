"""
====================================
Grid search + predictions
====================================
"""

import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
import interactionsmodel  # noqa for setting numba/numpy threads
from interactionsmodel.solvers import PGD, CD, CBPG_CS, CBPG_CS_mod, CBPG_permut
from interactionsmodel import path_data, path_tex
from interactionsmodel.utils import make_regression
from interactionsmodel.utils import get_lambda_max, kkt_violation
from sklearn.model_selection import train_test_split
import time
import json
from interactionsmodel.data_management import download_data


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


def classic_pgd(
    X, y, lambdas, device, full, init, instance, callback, use_acceleration=None
):
    if init:
        return PGD(X, y, lambdas, device, full, use_acceleration=use_acceleration)
    instance.run(MAXITER, eps=EPS, callback=callback)
    return instance.beta, instance.theta


def permut(
    X, y, lambdas, device, full, init, instance, callback, use_acceleration=None
):
    if init:
        return CBPG_permut(
            X, y, lambdas, device, full, use_acceleration=use_acceleration
        )
    instance.run(MAXITER, eps=EPS, callback=callback)
    return instance.beta, instance.theta


def cd(X, y, lambdas, device, full, init, instance, callback, use_acceleration=None):
    if init:
        return CD(X, y, lambdas)
    return instance.beta, instance.theta


def rando(X, y, lambdas, device, full, init, instance, callback, use_acceleration=None):
    if init:
        return CBPG_CS_mod(
            X, y, lambdas, device, full, use_acceleration=use_acceleration
        )
    instance.run(MAXITER, eps=EPS, callback=callback)
    return instance.beta, instance.theta


def cbpg_cs(
    X, y, lambdas, device, full, init, instance, callback, use_acceleration=None
):
    if init:
        return CBPG_CS(X, y, lambdas, device, full, use_acceleration=use_acceleration)
    instance.run(MAXITER, eps=EPS, callback=callback)
    return instance.beta, instance.theta


########################
# Prepare the benchmark
# ----------------------

# dtype in params
def prepare(
    X, y, penalties, use_torch=False, device=None, full=False, typefloat="double"
):
    # reproducibility
    np.random.seed(11235813)
    torch.manual_seed(11235813)
    if typefloat == "double":
        typedata = "float64"
    else:
        typedata = "float32"

    X_ = X.copy().astype(typedata)
    y_ = y.astype(typedata)
    pen = np.array(penalties).astype(typedata)
    if use_torch:
        X_tch = torch.from_numpy(X_).to(device)
        y_tch = torch.from_numpy(y_).clone().view(-1, 1).to(device)
        pen = torch.from_numpy(pen).to(device)
        return X_tch.contiguous(), y_tch, pen
    else:
        return X_, y_, pen


def bench_CD(instance, mse, times, X_test, y_test, db=None, dt=None):
    start = time.perf_counter()
    instance.run(MAXITER, eps=EPS)
    end = time.perf_counter() - start
    theta_i = instance.theta.flatten()
    beta_i = instance.beta.flatten()
    KKT = kkt_violation(instance, beta_i, theta_i)
    print(f"\t Iteration max {MAXITER} with KKT criterion {KKT}")

    times.append(end)
    if simulated:
        db.append(((beta - beta_i) ** 2).sum() / len(beta))
        dt.append(((theta - theta_i) ** 2).sum() / len(theta))

    X_test_ = X_test - instance.meanX
    X_test_ /= instance.stdX
    res = y_test.flatten() - X_test_ @ beta_i
    n, p = X_test.shape
    theta_prod = np.zeros((n,))
    jj = 0
    for j1 in range(p):
        for j2 in range(j1, p):
            Z_tmp = X_test[:, j1] * X_test[:, j2]
            Z_tmp = (Z_tmp - instance.meanZ[jj]) / instance.stdZ[jj]
            theta_prod += Z_tmp * theta_i[jj]
            jj += 1
    res -= theta_prod
    mse.append(np.linalg.norm(res, 2) ** 2 / len(res))


def bench_CBPG(
    instance, mse, times, X_test, y_test, db=None, dt=None, Li=None, LX=None, LZ=None
):
    if LX is None:
        torch.cuda.synchronize()
        start = time.perf_counter()
        Li, LX, LZ = instance.run(MAXITER, eps=EPS)
        torch.cuda.synchronize()
        end = time.perf_counter() - start
        theta_i = instance.theta
        beta_i = instance.beta
        KKT = kkt_violation(instance, beta_i, theta_i, bind="torch")
        print(
            f"\t Iteration max {MAXITER} with KKT criterion {KKT.cpu().item():.3f} stopped at {instance.cv}"
        )  # noqa
    else:
        torch.cuda.synchronize()
        start = time.perf_counter()
        instance.run(MAXITER, eps=EPS, Li=Li, LX=LX, LZ=LZ)
        torch.cuda.synchronize()
        end = time.perf_counter() - start
        theta_i = instance.theta
        beta_i = instance.beta
        KKT = kkt_violation(instance, beta_i, theta_i, bind="torch")
        print(
            f"\tIteration max {MAXITER} with KKT criterion {KKT.cpu().item():.3f} stopped at {instance.cv}"
        )  # noqa
    times.append(end)
    if simulated:
        db.append(
            (
                (torch.from_numpy(beta).float().to(device).view(-1, 1) - beta_i) ** 2
            ).sum()
            / len(beta)
        )
        dt.append(
            (
                (torch.from_numpy(theta).float().to(device).view(-1, 1) - theta_i) ** 2
            ).sum()
            / len(theta)
        )

    X_test_ = X_test - instance.meanX
    X_test_ /= instance.stdX
    res = y_test - X_test_ @ beta_i
    res -= instance.prod_Z(X_test, theta_i, instance.meanZ, instance.stdZ)
    res = res.flatten()
    mse.append(torch.linalg.norm(res, 2) ** 2 / len(res))
    if LX is not None:
        return Li, LX, LZ


def run_routine(
    X_, y_, lambdas, device, full, routine, acceleration, Li=None, LX=None, LZ=None
):
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, random_state=11235813)
    print(
        f"\tTest MSE with zero predictive power: {(y_test ** 2).sum() / len(y_test)}"
    )  # noqa
    instance = routine(
        X=X_train,
        y=y_train,
        lambdas=lambdas,
        device=device,
        full=full,
        init=True,
        instance=None,
        callback=None,
        use_acceleration=acceleration,
    )
    mse, times, dbeta, dtheta, callback = get_callback(instance)
    if routine == cd:
        bench_CD(instance, mse, times, X_test, y_test, dbeta, dtheta)
    elif LX is None:  # in CBPG and PGD
        Li, LX, LZ = bench_CBPG(instance, mse, times, X_test, y_test, dbeta, dtheta)
    else:
        bench_CBPG(instance, mse, times, X_test, y_test, dbeta, dtheta, Li, LX, LZ)
    return mse, times, dbeta, dtheta, Li, LX, LZ


def benchmark(X, y, l_lambdas, dic_args, datatype):
    l1 = []
    l2 = []
    mses = []
    times_ = []
    dbs, dts = [], []
    Li, LX, LZ = None, None, None
    for idx, lambdas in enumerate(l_lambdas):
        print(
            "\n",
            f"\r ##### Begin lambdas = ({lambdas[0]:.4f}, {lambdas[2]:.4f}), {idx+1}/{len(l_lambdas)}.",
        )  # noqa
        device_ = dic_args["device"]
        full = dic_args["full"]
        acceleration = dic_args["acceleration"]

        X_, y_, lambdas = prepare(
            X,
            y,
            lambdas,
            device=device_,
            use_torch=dic_args["use_torch"],
            typefloat=datatype,
        )

        mse, times, dbeta, dtheta, Li, LX, LZ = run_routine(
            X_,
            y_,
            lambdas,
            device_,
            full,
            dic_args["routine"],
            acceleration=acceleration,
            Li=Li,
            LX=LX,
            LZ=LZ,
        )
        if device_ == "cpu":
            l1.append(lambdas[0])
            l2.append(lambdas[2])
            mses.append(mse[0])
            times_.append(times[0])
            if simulated:
                dbs.append(dbeta[0])
                dts.append(dtheta[0])
        else:
            l1.append(lambdas[0].cpu().numpy().item())
            l2.append(lambdas[2].cpu().numpy().item())
            mses.append(mse[0].cpu().numpy().item())
            if simulated:
                dbs.append(dbeta[0].cpu().numpy().item())
                dts.append(dtheta[0].cpu().numpy().item())
            times_.append(times[0])
        print(f"\t Preliminary results: t={times_[-1]:.3f};MSE={mses[-1]:.3f}")
    return l1, l2, mses, times_, dbs, dts


def run_full_benchmark(routines_names, l_lambdas, datatype="double"):
    dict_ = {
        name: {
            "lambda_1": [],
            "lambda_2": [],
            "mse": [],
            "time": [],
            "dbs": [],
            "dts": [],
        }
        for name in routines_names
    }
    for routine in routines_names:
        print(
            f"\033[1;%dm #########" % 34 + "Begin " + routine + "#########\033[0m"
        )  # noqa
        print("~~~~~~ Beginning warmup")
        global MAXITER
        MAXITER = 1  # warmup

        dict_args = match_routine_args(routine)
        _ = benchmark(X, y, [l_lambdas[0]], dict_args, datatype)
        print("~~~~~~ Beginning benchmark")
        MAXITER = MAX_ITER
        dict_args = match_routine_args(routine)
        l1, l2, mse, time_, dbs, dts = benchmark(X, y, l_lambdas, dict_args, datatype)
        dict_[routine]["lambda_1"].append(l1)
        dict_[routine]["lambda_2"].append(l2)
        dict_[routine]["mse"].append(mse)
        dict_[routine]["time"].append(time_)
        dict_[routine]["dbs"].append(dbs)
        dict_[routine]["dts"].append(dts)
    return dict_


####################
# Make callback
# ------------------


def match_routine_args(routine_name):
    dict_args = {
        "PGD": {
            "routine": classic_pgd,
            "device": device,
            "use_torch": True,
            "full": False,
            "acceleration": False,
        },
        "CBPG": {
            "routine": cbpg_cs,
            "device": device,
            "use_torch": True,
            "full": False,
            "acceleration": False,
        },
        "PGD_nocuda": {
            "routine": classic_pgd,
            "device": "cpu",
            "use_torch": True,
            "full": False,
            "acceleration": False,
        },
        "CBPG_CS_nocuda": {
            "routine": cbpg_cs,
            "device": "cpu",
            "use_torch": True,
            "full": False,
            "acceleration": False,
        },
        "CD": {
            "routine": cd,
            "device": "cpu",
            "use_torch": False,
            "full": False,
            "acceleration": False,
        },
        "PGD_full": {
            "routine": classic_pgd,
            "device": device,
            "use_torch": True,
            "full": True,
            "acceleration": False,
        },
        "CBPG_CS_full": {
            "routine": cbpg_cs,
            "device": device,
            "use_torch": True,
            "full": True,
            "acceleration": False,
        },
        "PGD_acc": {
            "routine": classic_pgd,
            "device": device,
            "use_torch": True,
            "full": False,
            "acceleration": True,
        },
        "CBPG_acc": {
            "routine": cbpg_cs,
            "device": device,
            "use_torch": True,
            "full": False,
            "acceleration": True,
        },
        "rando_CBPG_acc": {
            "routine": rando,
            "device": device,
            "use_torch": True,
            "full": False,
            "acceleration": True,
        },
        "rando_CBPG": {
            "routine": rando,
            "device": device,
            "use_torch": True,
            "full": False,
            "acceleration": False,
        },
        "permut": {
            "routine": permut,
            "device": device,
            "use_torch": True,
            "full": False,
            "acceleration": False,
        },
        "permut_acc": {
            "routine": rando,
            "device": device,
            "use_torch": True,
            "full": False,
            "acceleration": True,
        },
    }
    return dict_args[routine_name]


def get_callback(instance):
    mse = []
    times = []
    dbs, dts = [], []

    def callback(it, beta_i, theta_i):
        reached_max = it != MAXITER
        return reached_max

    return mse, times, dbs, dts, callback


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
        print(f"MSE(beta) = {(beta ** 2).sum() / len(beta)}")
        print(f"MSE(theta) = {(theta ** 2).sum() / len(theta)}")

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
    X_ = X.copy()
    n_lambda = 10
    l1_ratio = 1
    title = "oban_genom_curve_paths"
    lambda_1, lambda_2 = get_lambda_max(X, y, bind="numpy", l1_ratio=l1_ratio)
    print(f"Lambda max are {lambda_1} and {lambda_2}")
    grid = np.logspace(np.log10(lambda_1), np.log10(lambda_1 / 100), n_lambda)
    l_alpha = [(alpha, alpha, lambda_1 * 20, lambda_1 * 20) for alpha in grid]
    print("~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~")

    routines_names = ["CBPG", "CD", "CBPG_acc"]
    dic_to_plot = run_full_benchmark(
        routines_names, l_lambdas=l_alpha, datatype="float"  # double to have 64
    )
    print("#### Plotting")
    for name in routines_names:
        dic_to_plot[name]["lambda_1"] = dic_to_plot[name]["lambda_1"][0]
        dic_to_plot[name]["lambda_2"] = dic_to_plot[name]["lambda_2"][0]
        dic_to_plot[name]["mse"] = dic_to_plot[name]["mse"][0]
        dic_to_plot[name]["time"] = dic_to_plot[name]["time"][0]
        if simulated:
            dic_to_plot[name]["dbs"] = dic_to_plot[name]["dbs"][0]
            dic_to_plot[name]["dts"] = dic_to_plot[name]["dts"][0]

    class Serializer(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            return json.JSONEncoder.default(self, obj)

    with open(os.path.join(path_data, title + ".json"), "w") as json_f:
        json.dump(dic_to_plot, json_f, cls=Serializer, indent=4)

    plt.figure()
    for name in routines_names:
        plt.plot(grid, dic_to_plot[name]["mse"], label=name)
    plt.xlabel("l1 regularization factor")
    plt.ylabel("MSE")
    # plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(path_save, title + "MSE.pdf"))  # noqa

    plt.figure()
    for name in routines_names:
        plt.plot(grid, dic_to_plot[name]["time"], label=name)
    plt.xlabel("l1 regularization factor")
    plt.ylabel("time (s)")
    # plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(path_save, title + "time.pdf"))  # noqa
    fig, ax = plt.subplots()
    for name in routines_names:
        ax.plot(
            grid,
            [
                dic_to_plot["CD"]["time"][i] / dic_to_plot[name]["time"][i]
                for i in range(n_lambda)
            ],
            linestyle="--" if name == "CD" else "solid",
            label=name,
        )
    ax.set_xlabel("l1 regularization factor")
    ax.set_ylabel("speedup against CD (ratio)")
    ax.yaxis.set_major_formatter("x{x:.1f}")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(path_save, title + "ratio.pdf"))  # noqa
    plt.show()

    if simulated:

        plt.figure()
        for name in routines_names:
            plt.plot(grid, dic_to_plot[name]["dbs"], label=name)
        plt.xlabel("l1 regularization factor")
        plt.ylabel(r"MSE(β_k - β*)")
        # plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig(os.path.join(path_save, title + "beta.pdf"))  # noqa

        plt.figure()
        for name in routines_names:
            plt.plot(grid, dic_to_plot[name]["dts"], label=name)
        plt.xlabel("l1 regularization factor")
        plt.ylabel(r"MSE(ϴ_k - ϴ*)")
        # plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig(os.path.join(path_save, title + "theta.pdf"))  # noqa
    plt.show()
