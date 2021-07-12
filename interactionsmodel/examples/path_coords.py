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
from interactionsmodel.solvers import PGD, CD, CBPG_CS, CBPG_CS_mod, CBPG_permut
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
MAX_ITER = int(100)
MAX_PM = int(1e1)
EPS_PM = 1e-3


def classic_pgd(
    X, y, lambdas, device, full, init, instance, callback, use_acceleration=None
):
    if init:
        return PGD(X, y, lambdas, device, full, use_acceleration=use_acceleration)
    instance.run(
        MAXITER, maxiter_power=MAX_PM, eps=EPS, eps_power=EPS_PM, callback=callback
    )
    return instance.beta, instance.theta


def permut(
    X, y, lambdas, device, full, init, instance, callback, use_acceleration=None
):
    if init:
        return CBPG_permut(
            X, y, lambdas, device, full, use_acceleration=use_acceleration
        )
    instance.run(
        MAXITER, maxiter_power=MAX_PM, eps=EPS, eps_power=EPS_PM, callback=callback
    )
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
    instance.run(
        MAXITER, eps=EPS, eps_power=EPS_PM, callback=callback, maxiter_power=MAX_PM
    )
    return instance.beta, instance.theta


def cbpg_cs(
    X, y, lambdas, device, full, init, instance, callback, use_acceleration=None
):
    if init:
        return CBPG_CS(X, y, lambdas, device, full, use_acceleration=use_acceleration)
    instance.run(MAXITER, eps=EPS, callback=callback, maxiter_power=MAX_PM)
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


def bench_CD(instance, its, times_, thetas, objs_, kkt_, gap_):
    stop = False
    next_val = 0
    while not stop:
        print(f"Going to iteration {next_val}")
        current_val = next_val
        t0 = time.perf_counter()
        instance.run(next_val, eps=EPS)
        end = time.perf_counter()
        theta_i = instance.theta
        beta_i = instance.beta
        _, _, gap, _, _ = instance.get_dual_gap(beta_i, theta_i)
        its.append(next_val)
        obj = instance.get_objective(beta_i, theta_i)
        times_.append(end - t0)
        objs_.append(obj)
        gap_.append(gap)
        kkt_.append(kkt_violation(instance, beta_i, theta_i))
        next_val = max(current_val + 1, min(int(1.5 * current_val), MAXITER))
        thetas.append(theta_i.reshape(-1)[begin_monitor:end_monitor])
        flat = np.isclose(kkt_[-2], kkt_[-1]) if len(kkt_) >= 2 else False
        if current_val == MAXITER or kkt_[-1] < EPS or flat:
            print("Ending")
            stop = True


def bench_CBPG(instance, its, times_, thetas, objs_, kkt_, gap_):
    stop = False
    next_val = 0
    while not stop:
        print(f"Going to iteration {next_val}")
        current_val = next_val
        t0 = time.perf_counter()
        instance.run(next_val, eps=EPS)
        end = time.perf_counter()
        theta_i = instance.theta
        beta_i = instance.beta
        _, _, gap, _, _ = instance.get_dual_gap(beta_i, theta_i)
        its.append(next_val)
        obj = instance.get_objective(beta_i, theta_i)
        times_.append(end - t0)
        objs_.append(obj)
        gap_.append(gap)
        kkt_.append(kkt_violation(instance, beta_i, theta_i, bind="torch"))
        next_val = max(current_val + 1, min(int(1.5 * current_val), MAXITER))
        thetas.append(theta_i.reshape(-1)[begin_monitor:end_monitor])
        flat = torch.isclose(kkt_[-2], kkt_[-1]) if len(kkt_) >= 2 else False
        if current_val == MAXITER or kkt_[-1] < EPS or flat:
            print("Ending")
            stop = True


def run_routine(X_, y_, lambdas, device, full, routine, acceleration):
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
    )
    its, times, thetas, objs, kkt, gap, callback = get_callback(instance)
    if routine == cd:
        bench_CD(instance, its, times, thetas, objs, kkt, gap)
    else:
        bench_CBPG(instance, its, times, thetas, objs, kkt, gap)
    return its, times, thetas, objs, kkt, gap


def benchmark(X, y, l_lambdas, dic_args, datatype):
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

        for n in range(1):
            its, times, thetas, objs, kkt, gaps = run_routine(
                X_,
                y_,
                lambdas,
                device_,
                full,
                dic_args["routine"],
                acceleration=acceleration,
            )
            print(f"Finished {n+1} out of {1}.")
        it_plot.append(its)
        time_plot.append(times)

        if device_ == "cpu":
            theta_plot.append([thetas[i] for i in range(len(thetas))])
            kkt_plot.append([kkt[i] for i in range(len(kkt))])
            obj_plot.append(objs)
            gap_plot.append(gaps)
        else:
            theta_plot.append([thetas[i].cpu() for i in range(len(thetas))])
            kkt_plot.append([kkt[i].cpu() for i in range(len(kkt))])
            obj_plot.append([obj.cpu().item() for obj in objs])
            gap_plot.append([gp.cpu().item() for gp in gaps])

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
            X, y, l_lambdas, dict_args, datatype
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

    def callback(it, beta_i, theta_i):
        if it == recorder["next_stop"]:
            t0 = time.perf_counter()
            recorder["delta_t"] += t0 - recorder["time"]

            print(f"Inside callback for iteration {it}.")

            its.append(it)
            if type(theta_i) is list:
                theta_temp = torch.cat(theta_i).flatten().clone()
            else:
                theta_temp = theta_i.flatten().clone()
            if end_monitor == -1:
                end_ = theta_temp.shape[0]
            else:
                end_ = end_monitor
            obj = instance.get_objective(beta_i, theta_temp.view(-1, 1))
            _, _, gap, _, _ = instance.get_dual_gap(beta_i, theta_temp.view(-1, 1))
            thetas.append(theta_temp[begin_monitor:end_])
            kkt_ = kkt_violation(
                instance, beta_i, theta_temp.reshape(-1, 1), bind="torch"
            )
            if instance.device == "cpu":
                gap = gap.item()

            print("kkt", it, "=", kkt_)
            kkt.append(kkt_)
            times_obj.append(recorder["delta_t"])
            values_obj.append(obj)
            gap_obj.append(gap)
            recorder["next_stop"] = min(next_stop_val(recorder["next_stop"]), MAXITER)

            # plateau
            flat = torch.isclose(kkt[-2], kkt[-1]) if len(kkt) >= 2 else False
            # iter
            reached_max = it != MAXITER
            # cv
            cv_ = kkt[-1] > EPS
            recorder["time"] = time.perf_counter()
            return reached_max and cv_ and not flat
        else:
            return True

    return its, times_obj, thetas, values_obj, kkt, gap_obj, callback


def next_stop_val(current):
    return max(current + 1, min(int(1.5 * current), MAXITER))


if __name__ == "__main__":
    save_fig = True
    begin_monitor = 0
    end_monitor = 1
    #############################
    # Load the data
    #############################

    print(f"Cuda is {used_cuda}.")
    snr = 10
    n_samples, n_features = 20000, 500
    X, y, beta0, beta, theta = make_regression(
        n_samples,
        n_features,
        False,
        mu=0,
        sigma=1,
        seed=112358,
        snr=snr,
        sparse_inter=0.25,
        sparse_main=0.25,
    )
    X_ = X.copy()
    lambda_1, lambda_2 = get_lambda_max(X_, y, bind="numpy")
    lambda_1 = max(lambda_1, lambda_2)
    lambda_1 /= 10
    lambda_2 = lambda_1 / 10
    l_alpha = [(lambda_1, lambda_1, lambda_2, lambda_2)]
    obj_y = 0.5 * 1 / X.shape[0] * np.linalg.norm(y, 2) ** 2
    print(f"1/2n ||y||^2 = {obj_y :.3f}.")

    print("~~~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~~~")

    routines_names = ["CBPG", "CD", "CBPG_acc", "PGD", "PGD_acc"]
    dic_to_plot = run_full_benchmark(
        routines_names, l_lambdas=l_alpha, datatype="float"  # double to have 64
    )
    print("plotting")
    for name in routines_names:
        dic_to_plot[name]["its"] = dic_to_plot[name]["its"][0][0]
        dic_to_plot[name]["thetas"] = dic_to_plot[name]["thetas"][0][0]
        dic_to_plot[name]["kkt"] = dic_to_plot[name]["kkt"][0][0]
        dic_to_plot[name]["gaps"] = dic_to_plot[name]["gaps"][0][0]
        dic_to_plot[name]["times"] = dic_to_plot[name]["times"][0][0]
        dic_to_plot[name]["objs"] = dic_to_plot[name]["objs"][0][0]

    plt.figure()
    for name in routines_names:
        plt.plot(dic_to_plot[name]["times"], dic_to_plot[name]["kkt"], label=name)
    plt.ylabel("KKT violation")
    plt.xlabel("Time (s)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            os.path.join(
                path_save,
                f"simulated_n{n_samples}p{n_features}_kkt_snr{snr}_over100_sp_32.pdf",
            )
        )  # noqa

    plt.figure()
    for name in routines_names:
        plt.plot(dic_to_plot[name]["times"], dic_to_plot[name]["objs"], label=name)
    plt.ylabel("Objective")
    plt.xlabel("Time (s)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            os.path.join(
                path_save,
                f"simulated_n{n_samples}p{n_features}_snr{snr}_over100_sp_32.pdf",
            )
        )  # noqa
    plt.show()
