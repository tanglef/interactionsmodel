import numpy as np
import torch
import os
import seaborn as sns
import time
import matplotlib.pyplot as plt
from interactionsmodel import path_data, path_tex
from interactionsmodel.data_management import download_data
from interactionsmodel.solvers import PGD, CBPG_CS, CD, CBPG_CS_mod
from interactionsmodel.utils import get_lambda_max, kkt_violation

plt.rcParams.update({"font.size": 16})
sns.set()


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
dtype = torch.float64
device = "cuda" if use_cuda else "cpu"
used_cuda = "used" if use_cuda else "not used"
MAX_ITER = int(200)
EPS = 1e-3
EPS_PM = 1e-7
MAX_PM = int(1e3)

##########################
# Functions to Benchmark
# ------------------------


def classic_pgd(
    X, y, lambdas, device, full, init, instance, callback, use_acceleration=None
):
    if init:
        return PGD(X, y, lambdas, device, full, use_acceleration=use_acceleration)
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


def prepare(X, y, pen, typefloat="float32", use_torch=False, device="cuda"):
    # reproducibility
    np.random.seed(11235813)
    torch.manual_seed(11235813)

    X_ = X.copy().astype(typefloat)
    pen = np.array(pen).astype(typefloat)
    y_ = y.astype(typefloat)
    print("-------------Finished computing means and sd")
    if use_torch:
        X_tch = torch.from_numpy(X_).to(device)
        y_tch = torch.from_numpy(y_).clone().view(-1, 1).to(device)
        return X_tch.contiguous(), y_tch, torch.from_numpy(pen).to(device)
    else:
        return X_, y_, pen


def bench_CD(instance, its, times_, thetas, objs_, kkt_, gap_):
    stop = False
    next_val = 0
    while not stop:
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
        print(
            f"Iteration {current_val}. Sparsity is {np.linalg.norm(theta_i.reshape(-1), 0) / theta_i.shape[0]:.3f}"
        )  # noqa
        next_val = max(current_val + 1, min(int(1.5 * current_val), MAXITER))
        flat = np.isclose(kkt_[-2], kkt_[-1]) if len(kkt_) >= 2 else False
        if current_val == MAXITER or kkt_[-1] < EPS or flat:
            print("Ending")
            stop = True


def bench_CBPG(instance, its, times_, thetas, objs_, kkt_, gap_):
    stop = False
    next_val = 0
    while not stop:
        current_val = next_val
        t0 = time.perf_counter()
        instance.run(next_val, eps=EPS)
        end = time.perf_counter()
        theta_i = instance.theta
        beta_i = instance.beta
        if type(theta_i) is list:
            theta_i = torch.cat(theta_i).view(-1, 1).clone()
        _, _, gap, _, _ = instance.get_dual_gap(beta_i, theta_i)
        its.append(next_val)
        obj = instance.get_objective(beta_i, theta_i)
        times_.append(end - t0)
        objs_.append(obj)
        gap_.append(gap)
        kkt_.append(kkt_violation(instance, beta_i, theta_i, bind="torch"))
        next_val = max(current_val + 1, min(int(1.5 * current_val), MAXITER))
        print(
            f"Iteration {current_val}. Sparsity is {torch.linalg.norm(theta_i.reshape(-1), 0) / theta_i.shape[0]:.3f}"
        )  # noqa
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
        "PGD_nocuda": {
            "routine": classic_pgd,
            "device": "cpu",
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
        "CBPG_nocuda": {
            "routine": cbpg_cs,
            "device": "cpu",
            "use_torch": True,
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
            obj = instance.get_objective(beta_i, theta_temp.view(-1, 1))
            _, _, gap, _, _ = instance.get_dual_gap(beta_i, theta_temp.view(-1, 1))
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
    save_fig = False
    fast_data = False  # take only 10 features
    MAXITER = 1  # the warmup
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
    y = y[:, 0]
    obj_y = 0.5 * 1 / X.shape[0] * np.linalg.norm(y, 2) ** 2
    print(
        f"The 1/2n * sqnorm of y is \
    {obj_y :.3f}."
    )
    lambda_1, lambda_2 = get_lambda_max(X, y, bind="numpy")
    lambda_1 /= 5000
    lambda_2 /= 5000
    l_alpha = [(lambda_1, lambda_1, lambda_2, lambda_2)]
    print("~~~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~~~")

    #######################
    # Benchmark
    #######################
    routines_names = ["CBPG", "CD", "CBPG_acc", "PGD", "PGD_acc", "rando_CBPG"]
    # routines_names = ["PGD_acc", "CBPG_CS_acc",
    #                   "CBPG_CS", "PGD",
    #                   # "PGD_full", "CBPG_CS_full",
    #                   # "CD" ]
    #                   "CBPG_CS_nocuda", "PGD_nocuda"]

    dic_to_plot = run_full_benchmark(
        routines_names, l_lambdas=l_alpha, datatype="float32"
    )
    for name in routines_names:
        dic_to_plot[name]["times"] = dic_to_plot[name]["times"][0][0]
        dic_to_plot[name]["objs"] = dic_to_plot[name]["objs"][0][0]
        dic_to_plot[name]["kkt"] = dic_to_plot[name]["kkt"][0][0]
        # dic_to_plot[name]["gaps"] = dic_to_plot[name]["gaps"][0][0]

    fig, ax = plt.subplots()
    for idx, name in enumerate(routines_names):
        ax.plot(dic_to_plot[name]["times"], dic_to_plot[name]["objs"], label=name)
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
        fig.savefig(os.path.join(path_save, "benchmark_cd_cbpg_genom.pdf"))
    plt.show(block=False)

    plt.figure()
    for idx, name in enumerate(routines_names):
        plt.plot(dic_to_plot[name]["times"], dic_to_plot[name]["kkt"], label=name)
    plt.ylabel("KKT violation")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(path_save, "curve_kkt_cd_cbpg_genom.pdf"))
    plt.show()
