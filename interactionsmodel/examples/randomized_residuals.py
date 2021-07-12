"""
=================================================
Application of the PGD/CD/CBPG to simulated dataset
=================================================

This includes:
    - a benchmark of precision vs time,
    - the final dual gap,
"""

import numpy as np
import torch
import os
import seaborn as sns
import time
import matplotlib.pyplot as plt
from interactionsmodel.solvers import CBPG_CS_mod, CD, CBPG_CS
from interactionsmodel import path_tex

# from sklearn.datasets import make_regression
from interactionsmodel.utils import get_lambda_max, make_regression


plt.rcParams.update({"font.size": 16})
sns.set()


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
EPS = 1e-10
MAX_ITER = int(300)
EPS_PM = 1e-15
seed = 11235813


##########################
# Functions to Benchmark
# ------------------------


def rando(X, y, lambdas, device, full, init, instance, callback, use_acceleration=None):
    if init:
        return CBPG_CS_mod(
            X, y, lambdas, device, full, use_acceleration=use_acceleration
        )
    instance.run(MAXITER, eps=EPS, eps_power=EPS_PM, callback=callback)
    return instance.beta, instance.theta


def cbpg_cs(
    X, y, lambdas, device, full, init, instance, callback, use_acceleration=None
):
    if init:
        return CBPG_CS(X, y, lambdas, device, full, use_acceleration=use_acceleration)
    instance.run(MAXITER, eps=EPS, callback=callback)
    return instance.beta, instance.theta


def cd(X, y, lambdas, device, full, init, instance, callback, use_acceleration=None):
    if init:
        return CD(X, y, lambdas)
    return instance.beta, instance.theta


########################
# Prepare the benchmark
# ----------------------


def prepare(X, y, penalties, use_torch=False, device=None, full=False):
    # reproducibility
    np.random.seed(11235813)
    torch.manual_seed(11235813)
    X_ = X.copy()
    pen = np.array(penalties).astype("float64")
    if use_torch:
        X_tch = torch.from_numpy(X_).to(device)
        y_tch = torch.from_numpy(y).clone().view(-1, 1).to(device)
        return X_tch.contiguous(), y_tch, torch.from_numpy(pen).to(device)
    else:
        X_ = X_.astype("float64")
        y_ = y.astype("float64")
        return X_, y_, pen


def bench_CD(instance, its, times_, objs_, gap_):
    stop = False
    next_val = 0
    while not stop:
        print(f"Going to iteration {next_val}")
        its.append(next_val)
        current_val = next_val
        t0 = time.perf_counter()
        instance.run(next_val, eps=EPS)
        end = time.perf_counter()
        beta_i, theta_i = instance.beta, instance.theta
        obj = instance.get_objective(beta_i, theta_i)
        _, _, gap, _, _ = instance.get_dual_gap(beta_i, theta_i)
        times_.append(end - t0)
        objs_.append(obj)
        gap_.append(gap)
        next_val = min(next_stop_val(next_val), MAXITER)
        if current_val == MAXITER or gap < EPS:
            stop = True
    return its, times_, objs_, gap_


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
    its, times_, objs_, gap_, callback = get_callback(instance)
    if routine != cd:
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
    else:
        bench_CD(instance, its, times_, objs_, gap_)
    return times_, objs_, gap_


def benchmark(X, y, l_lambdas, dic_args, n_repeat=10):
    time_plot = []
    obj_plot = []
    gap_plot = []
    for idx, lambdas in enumerate(l_lambdas):
        print(
            "\n",
            f"\r ##### Begin lambdas = ({lambdas[0]:.4f}, {lambdas[1]:.4f}), {idx+1}/{len(l_lambdas)}.",
        )  # noqa
        times_fixed = []
        objs_fixed = []
        gaps_fixed = []
        device_ = dic_args["device"]
        full = dic_args["full"]
        acceleration = dic_args["acceleration"]

        X_, y_, lambdas = prepare(
            X, y, lambdas, device=device_, use_torch=dic_args["use_torch"]
        )

        for n in range(n_repeat):
            times_, objs_, gaps = run_routine(
                X_,
                y_,
                lambdas,
                device_,
                full,
                dic_args["routine"],
                acceleration=acceleration,
            )
            times_fixed.append(times_)
            objs_fixed.append(objs_)
            gaps_fixed.append(gaps)
            print(f"Finished {n+1} out of {n_repeat}.")
        time_plot.append(
            [
                np.median([tt[i] for tt in times_fixed], axis=0)
                for i in range(len(times_fixed[0]))
            ]
        )
        if device_ == "cpu":
            obj_plot.append(
                [
                    np.median([obj[i].item() for obj in objs_fixed], axis=0)
                    for i in range(len(objs_fixed[0]))
                ]
            )
            gap_plot.append(
                [
                    np.median([gp[i] for gp in gaps_fixed], axis=0)
                    for i in range(len(gaps_fixed[0]))
                ]
            )
        else:
            obj_plot.append(
                [
                    np.median([obj[i].cpu().item() for obj in objs_fixed], axis=0)
                    for i in range(len(objs_fixed[0]))
                ]
            )
            gap_plot.append(
                [
                    np.median([gp[i].cpu().item() for gp in gaps_fixed], axis=0)
                    for i in range(len(gaps_fixed[0]))
                ]
            )
    return time_plot, obj_plot, gap_plot


def run_full_benchmark(routines_names, l_lambdas, n_repeat=20):
    dict_ = {name: {"times": [], "objs": [], "gaps": []} for name in routines_names}
    for routine in routines_names:
        global MAXITER
        MAXITER = 1  # warmup

        print(
            f"\033[1;%dm #########" % 34 + "Begin " + routine + "#########\033[0m"
        )  # noqa
        dict_args = match_routine_args(routine)
        times, objs, gaps = benchmark(X, y, l_lambdas, dict_args, n_repeat=1)
        print("~~~~~~~~ Warmup finished. Beginning benchmark")
        MAXITER = MAX_ITER
        dict_args = match_routine_args(routine)
        times, objs, gaps = benchmark(X, y, l_lambdas, dict_args, n_repeat=n_repeat)
        dict_[routine]["times"].append(times)
        dict_[routine]["objs"].append(objs)
        dict_[routine]["gaps"].append(gaps)
    return dict_


####################
# Make callback
# ------------------


def match_routine_args(routine_name):
    dict_args = {
        "rando_CBPG": {
            "routine": rando,
            "device": device,
            "use_torch": True,
            "full": False,
            "acceleration": True,
        },
        "CBPG_CS": {
            "routine": cbpg_cs,
            "device": device,
            "use_torch": True,
            "full": False,
            "acceleration": True,
        },
        "CD": {
            "routine": cd,
            "device": "cpu",
            "use_torch": False,
            "full": False,
            "acceleration": False,
        },
    }
    return dict_args[routine_name]


def get_callback(instance):
    global obj_y
    if instance.device == "cpu":
        primal = obj_y
    else:
        primal = torch.tensor(obj_y, device=instance.device, dtype=torch.float64)
    values_obj = [primal]
    its = [0]
    gap_obj = [primal]
    times_obj = [0]
    recorder = {"time": time.perf_counter(), "delta_t": 0, "next_stop": 0}

    def callback(it, beta_i, theta_i):
        if it == recorder["next_stop"]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            recorder["delta_t"] += t0 - recorder["time"]
            print(f"Inside callback for iteration {it}.")
            its.append(it)
            if isinstance(theta_i, list):
                chg = torch.cat(theta_i)
            else:
                chg = theta_i
            obj = instance.get_objective(beta_i, chg)
            _, _, gap, _, _ = instance.get_dual_gap(beta_i, chg)
            if instance.device == "cpu":
                gap = gap.item()
            times_obj.append(recorder["delta_t"])
            values_obj.append(obj)
            gap_obj.append(gap)
            recorder["next_stop"] = min(next_stop_val(recorder["next_stop"]), MAXITER)
            recorder["time"] = time.perf_counter()
        return it != MAXITER and gap_obj[-1] > EPS

    return its, times_obj, values_obj, gap_obj, callback


def next_stop_val(current):
    return max(current + 1, min(int(1.5 * current), MAXITER))


if __name__ == "__main__":
    save_fig = False
    #############################
    # Load the data
    #############################

    print(f"Cuda is {used_cuda}.")

    # get the data
    # X, y = make_regression(n_samples, n_features, n_informative=20,
    #                        random_state=11235813)
    n_samples, n_features = 100, 50
    X, y, beta0, beta, theta = make_regression(
        n_samples, n_features, False, mu=0, sigma=1, seed=112358
    )
    # inter_only = False
    # corr = 0.9
    # beta_sparsity, theta_sparsity = 5, 5
    # choice_features = np.array([-10, 10])
    # corr_expected = np.zeros((n_features, n_features))
    # for a in range(n_features):
    #     for i in range(n_features):
    #         corr_expected[a, i] = corr**abs(a-i)

    # X = multivariate_normal(mean=np.arange(n_features),
    #                         cov=corr_expected,
    #                         size=n_samples)
    # X_cpt_Z = np.copy(X, order="F")
    # (y, beta, theta, sigma,
    #     noise) = make_data(X, inter_only,
    #                        1,  # scaled gaussian
    #                        beta_sparsity,
    #                        theta_sparsity, choice_features,
    #                        True,
    #                        seed=seed)
    X_ = X.copy()
    lambda_1_max, lambda_2_max = get_lambda_max(X_, y)
    lambda_1_max, lambda_2_max = lambda_1_max / 10, lambda_2_max / 10
    lambdas = [(lambda_1_max, lambda_1_max, lambda_2_max, lambda_2_max)]
    print(f"Using lambda1 = {lambda_1_max:.3f}," + f"lambda2 = {lambda_2_max:.3f}.")
    obj_y = 0.5 * 1 / X.shape[0] * np.linalg.norm(y, 2) ** 2
    print(f"1/2n ||y||^2 = {obj_y :.3f}.")
    print("~~~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~~~")

    #######################
    # Benchmark
    #######################

    # warmup

    routines_names = ["rando_CBPG", "CBPG_CS", "CD"]
    dic_to_plot = run_full_benchmark(routines_names, l_lambdas=lambdas, n_repeat=1)

    for name in routines_names:
        dic_to_plot[name]["times"] = dic_to_plot[name]["times"][0][0]
        dic_to_plot[name]["gaps"] = dic_to_plot[name]["gaps"][0][0]
        dic_to_plot[name]["objs"] = dic_to_plot[name]["objs"][0][0]

    fig, ax = plt.subplots()
    markers = ["*", "+", "^"]
    for idx, name in enumerate(routines_names):
        ax.plot(
            dic_to_plot[name]["times"],
            dic_to_plot[name]["objs"],
            label=name,
            marker=markers[idx],
        )
    ax.set_ylabel("Objective value")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.set_yscale("log")
    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    fig.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(path_save, "simulated_solvers_rando.pdf"))
    plt.show(block=False)

    plt.figure()
    for idx, name in enumerate(routines_names):
        plt.plot(
            dic_to_plot[name]["times"],
            dic_to_plot[name]["gaps"],
            label=name,
            marker=markers[idx],
        )
    plt.ylabel("Dual gap")
    plt.xlabel("Time (s)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(path_save, "gaps_simulated_rando.pdf"))
    plt.show()
