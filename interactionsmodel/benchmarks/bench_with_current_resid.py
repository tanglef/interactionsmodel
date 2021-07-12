"""
====================================
Look where the coordinates a block
====================================
"""

import numpy as np
import torch
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from interactionsmodel.solvers import CBPG_CS
from interactionsmodel import path_data, path_tex
from interactionsmodel.utils import make_regression

# from numpy.random import multivariate_normal
from interactionsmodel.utils import get_lambda_max, kkt_violation


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
dtype = torch.float64
device = "cuda" if use_cuda else "cpu"
used_cuda = "used" if use_cuda else "not used"
EPS = 1e-3
MAX_ITER = int(6)
MAX_PM = int(1e1)
EPS_PM = 1e-3


def cbpg_cs(
    X,
    y,
    lambdas,
    device,
    full,
    init,
    instance,
    callback,
    use_acceleration=None,
    benchmark=True,
):
    if init:
        return CBPG_CS(
            X,
            y,
            lambdas,
            device,
            full,
            use_acceleration=use_acceleration,
            benchmark=benchmark,
        )
    instance.run(
        MAXITER,
        eps=EPS,
        callback=callback,
        maxiter_power=MAX_PM,
    )
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


def run_routine(X_, y_, lambdas, device, full, routine, acceleration, benchmark):
    instance = routine(
        X=X_,
        y=y_,
        lambdas=lambdas,
        device=device,
        full=full,
        init=True,
        instance=None,
        callback=None,
        use_acceleration=acceleration,
        benchmark=benchmark,
    )
    its, times, thetas, objs, kkt, gap, callback = get_callback(instance)
    routine(
        X=None,
        y=None,
        lambdas=None,
        device=None,
        full=None,
        instance=instance,
        init=False,
        callback=callback,
    )
    return its, times, thetas, objs, kkt, gap


def benchmark(X, y, l_lambdas, dic_args, datatype, rep=1):
    it_plot = []
    kkt_plot = []
    obj_plot = []
    theta_plot = []
    gap_plot = []
    time_plot = []
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
        times_f = []
        kkt_f = []
        for n in range(rep):
            its, times, thetas, objs, kkt, gaps = run_routine(
                X_,
                y_,
                lambdas,
                device_,
                full,
                dic_args["routine"],
                acceleration=acceleration,
                benchmark=dic_args["benchmark"],
            )
            times_f.append(times)
            kkt_f.append(kkt)
            print(f"Finished {n+1} out of {rep}.")
        it_plot.append(its)
        time_plot.append(
            [
                np.median([tt[i] for tt in times_f], axis=0)
                for i in range(len(times_f[0]))
            ]
        )
        kkt_plot.append(
            [
                np.median([kkt_[i].cpu().numpy() for kkt_ in kkt_f], axis=0)
                for i in range(len(kkt_f[0]))
            ]
        )

    return it_plot, theta_plot, kkt_plot, time_plot, gap_plot, obj_plot


def run_full_benchmark(routines_names, l_lambdas, datatype="double"):
    dict_ = {
        name: {"its": [], "thetas": [], "kkt": [], "objs": [], "times": [], "gaps": []}
        for name in routines_names
    }
    for routine in routines_names:

        print(
            f"\033[1;%dm #########" % 34 + "Begin " + routine + "#########\033[0m"
        )  # noqa
        global MAXITER
        MAXITER = 1  # warmup

        dict_args = match_routine_args(routine)
        _ = benchmark(X, y, l_lambdas, dict_args, datatype)
        print("~~~~~~~~ Warmup finished. Beginning benchmark")
        MAXITER = MAX_ITER
        dict_args = match_routine_args(routine)
        its, thetas, kkt, times, gaps, objs = benchmark(
            X, y, l_lambdas, dict_args, datatype, rep=5
        )
        dict_[routine]["its"].append(its)
        dict_[routine]["thetas"].append(thetas)
        dict_[routine]["kkt"].append(kkt)
        dict_[routine]["gaps"].append(gaps)
        dict_[routine]["times"].append(times)
        dict_[routine]["objs"].append(objs)
    return dict_


####################
# Make callback
# ------------------


def match_routine_args(routine_name):
    dict_args = {
        "CBPG_CS": {
            "routine": cbpg_cs,
            "device": device,
            "use_torch": True,
            "full": False,
            "acceleration": False,
            "benchmark": True,
        },
        "CBPG_CS_current_resid": {
            "routine": cbpg_cs,
            "device": device,
            "use_torch": True,
            "full": False,
            "acceleration": False,
            "benchmark": False,
        },
    }
    return dict_args[routine_name]


def get_callback(instance):
    global obj_y
    if instance.device == "cpu":
        primal = obj_y
    else:
        primal = torch.tensor(obj_y, device=instance.device, dtype=instance.X.dtype)

    recorder = {"time": time.perf_counter(), "delta_t": 0, "next_stop": 0}
    values_obj = [primal]
    its = [0]
    thetas = []
    gap_obj = [primal]
    times_obj = [0]
    kkt = [primal]

    def callback(it, beta_i, theta_i, resid=None):
        if it == recorder["next_stop"]:
            its.append(it)
            theta_temp = theta_i
            if instance.benchmark:
                kkt_ = kkt_violation(instance, beta_i, theta_temp, bind="torch")
            else:
                kkt_ = kkt_violation(
                    instance, beta_i, theta_temp, bind="torch", resid=-resid
                )
            kkt.append(kkt_)
            recorder["next_stop"] += 1
            times_obj.append(time.perf_counter() - recorder["time"])
        return kkt[-1] > EPS and it != MAXITER

    return its, times_obj, thetas, values_obj, kkt, gap_obj, callback


if __name__ == "__main__":
    save_fig = False
    begin_monitor = 0
    end_monitor = 1
    #############################
    # Load the data
    #############################

    print(f"Cuda is {used_cuda}.")

    n_samples, n_features = 15000, 500
    X, y, beta0, beta, theta = make_regression(
        n_samples, n_features, False, mu=0, sigma=1, seed=112358
    )
    X_ = X.copy()
    lambda_1, lambda_2 = get_lambda_max(X_, y, bind="numpy")
    lambda_1 /= 10
    lambda_2 /= 10
    l_alpha = [(lambda_1, lambda_1, lambda_2, lambda_2)]
    obj_y = 0.5 * 1 / X.shape[0] * np.linalg.norm(y, 2) ** 2
    print(f"1/2n ||y||^2 = {obj_y :.3f}.")

    print("~~~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~~~")

    routines_names = ["CBPG_CS", "CBPG_CS_current_resid"]
    dic_to_plot = run_full_benchmark(
        routines_names, l_lambdas=l_alpha, datatype="float"
    )
    print("plotting")
    for name in routines_names:
        dic_to_plot[name]["its"] = dic_to_plot[name]["its"][0][0]
        dic_to_plot[name]["kkt"] = dic_to_plot[name]["kkt"][0][0]
        dic_to_plot[name]["times"] = dic_to_plot[name]["times"][0][0]

    plt.figure()
    for name in routines_names:
        plt.plot(dic_to_plot[name]["times"], dic_to_plot[name]["kkt"], label=name)
    plt.ylabel("distance to 0 for subdifferential")  # in maths!!!
    plt.xlabel("Time (s)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            os.path.join(
                path_save, f"curent_vs_compute_resids_{n_samples}p{n_features}.pdf"
            )
        )  # noqa
    plt.show()
