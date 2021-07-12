import numpy as np
import torch
import os
import seaborn as sns
import time
import matplotlib.pyplot as plt
from interactionsmodel import path_data, path_tex
from interactionsmodel.data_management import download_data
from interactionsmodel.solvers import CD
from interactionsmodel.utils import get_lambda_max


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
dtype = torch.float32
device = "cuda" if use_cuda else "cpu"
used_cuda = "used" if use_cuda else "not used"
MAX_ITER = int(1e3)
EPS = 1e-7
EPS_PM = 1e-7


##########################
# Functions to Benchmark
# ------------------------


def enet_cd(
    X=None,
    y=None,
    alphas=None,
    device=None,
    full=None,
    init=False,
    instance=None,
    callback=None,
    use_acceleration=None,
    maxiter=None,
):
    if init:
        return CD(X, y, alphas)
    instance.run(maxiter, eps=EPS, callback=callback)
    return instance.beta, instance.theta


########################
# Prepare the benchmark
# ----------------------


def prepare(X, y, use_torch=False, device="cuda", full=False):
    # reproducibility
    np.random.seed(11235813)
    torch.manual_seed(11235813)

    X_ = X.copy()
    X_ = X_.astype("float32")
    y_ = y.astype("float32")
    return X_, y_


def bench_CD(instance, times_, objs_, gap_):
    stop = False
    next_val = 0
    while not stop:
        print(next_val)
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
        if gap < 1e-5 or next_val == current_val:
            stop = True
    return times_, objs_, gap_


def run_routine(X_, y_, alphas, device, full, acceleration, routine):
    instance = routine(
        X=X_,
        y=y_,
        alphas=alphas,
        device=device,
        full=full,
        init=True,
        use_acceleration=acceleration,
    )
    times_, objs_, gap_, callback = get_callback(instance)
    bench_CD(instance, times_, objs_, gap_)
    return times_, objs_, gap_


def benchmark(X, y, l_alphas, dic_args, n_repeat=10):
    time_plot = []
    obj_plot = []
    gap_plot = []
    for idx, alphas in enumerate(l_alphas):
        print(
            "\n",
            f"\r ##### Begin alphas = ({alphas[0]:.4f}, {alphas[1]:.4f}), {idx+1}/{len(l_alphas)}.",
        )  # noqa
        times_fixed = []
        objs_fixed = []
        gaps_fixed = []
        device_ = dic_args["device"]
        full = dic_args["full"]
        use_torch = dic_args["use_torch"]
        acceleration = dic_args["acceleration"]
        X_, y_ = prepare(X, y, device=device_, use_torch=use_torch)

        for n in range(n_repeat):
            times_, objs_, gaps_ = run_routine(
                X_, y_, alphas, device_, full, acceleration, dic_args["routine"]
            )
            times_fixed.append(times_)
            objs_fixed.append(objs_)
            gaps_fixed.append(gaps_)
            print(f"Finished {n+1} out of {n_repeat}.")
        time_plot.append(
            [
                np.median([tt[i] for tt in times_fixed], axis=0)
                for i in range(len(times_fixed[0]))
            ]
        )
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
    return time_plot, obj_plot, gap_plot


def run_full_benchmark(routines_names, l_alphas, n_repeat=20):
    dict_ = {name: {"times": [], "objs": [], "gaps": []} for name in routines_names}
    for routine in routines_names:
        print(
            f"\033[1;%dm #########" % 34 + "Begin " + routine + "#########\033[0m"
        )  # noqa
        dict_args = match_routine_args(routine)
        times, objs, gaps = benchmark(X, y, l_alphas, dict_args, n_repeat=1)
        print("~~~~~~~~ Warmup finished. Beginning benchmark")
        global MAXITER
        MAXITER = MAX_ITER
        dict_args = match_routine_args(routine)
        times, objs, gaps = benchmark(X, y, l_alphas, dict_args, n_repeat=n_repeat)
        dict_[routine]["times"].append(times)
        dict_[routine]["objs"].append(objs)
        dict_[routine]["gaps"].append(gaps)
    return dict_


####################
# Make callback
# ------------------


def match_routine_args(routine_name):
    dict_args = {
        "CD": {
            "routine": enet_cd,
            "device": "cpu",
            "use_torch": False,
            "full": False,
            "acceleration": False,
        }
    }
    return dict_args[routine_name]


def get_callback(instance):
    global obj_y
    if instance.device == "cpu":
        primal = obj_y
    else:
        primal = torch.tensor(obj_y, device=instance.device, dtype=torch.float32)
    values_obj = [primal]
    times_obj = [0]
    gap_obj = [primal]
    recorder = {"time": time.perf_counter(), "delta_t": 0, "next_stop": 0}

    def callback(it, beta_i, theta_i):
        if it == recorder["next_stop"]:
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            print(f"Inside callback for iteration {it}.")
            recorder["delta_t"] += t0 - recorder["time"]
            obj = instance.get_objective(beta_i, theta_i)
            _, _, gap, _, _ = instance.get_dual_gap(beta_i, theta_i)
            if instance.device == "cpu":
                gap = gap.item()
            times_obj.append(recorder["delta_t"])
            values_obj.append(obj)
            gap_obj.append(gap)
            recorder["next_stop"] = min(next_stop_val(recorder["next_stop"]), MAXITER)
            if use_cuda:
                torch.cuda.synchronize()

            recorder["time"] = time.perf_counter()

    return times_obj, values_obj, gap_obj, callback


def next_stop_val(current):
    return max(current + 1, min(int(1.5 * current), MAXITER))


if __name__ == "__main__":
    save_fig = False
    fast_data = True  # take only 10 features
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
    lambda_1 /= 10
    lambda_2 /= 10
    l_alpha = [(lambda_1, lambda_1, lambda_2, lambda_2)]
    print("~~~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~~~")

    #######################
    # Benchmark
    #######################

    routines_names = ["CD"]
    dic_to_plot0 = run_full_benchmark(routines_names, l_alphas=l_alpha, n_repeat=1)
    dic_to_plot1 = run_full_benchmark(routines_names, l_alphas=l_alpha, n_repeat=2)
    dic_to_plot2 = run_full_benchmark(routines_names, l_alphas=l_alpha, n_repeat=3)

    plt.figure()
    for idx, name in enumerate(routines_names):
        plt.plot(
            dic_to_plot0[name]["times"][0][0],
            dic_to_plot0[name]["objs"][0][0],
            label="0",
        )
        plt.plot(
            dic_to_plot1[name]["times"][0][0],
            dic_to_plot1[name]["objs"][0][0],
            label="1",
        )
        plt.plot(
            dic_to_plot2[name]["times"][0][0],
            dic_to_plot2[name]["objs"][0][0],
            label="2",
        )

    plt.ylabel("Objective value")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(path_save, "benchmark_solvers.pdf"))
    plt.show()
