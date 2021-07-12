import numpy as np
import torch
import os
import seaborn as sns
import time
import matplotlib.pyplot as plt
from interactionsmodel.utils import cpt_mean_std, power_method, Lanczos
from interactionsmodel import path_tex
from scipy.sparse.linalg import svds

plt.rcParams.update({"font.size": 16})
save_fig = True
sns.set()

############################################################################
# Debugging
# (clean_pykeops if Core dumped at previous trial can be useful)
############################################################################

# import pykeops
# pykeops.clean_pykeops()
# pykeops.test__numpy_bindings()
# pykeops.clean_pykeops()
# pykeops.test_torch_bindings()
# pykeops.config.verbose = True
# pykeops.config.build_type = 'Debug'

##########################
# Load and prepare data
###########################

# path to the data
path_file = os.path.dirname(__file__)
path_save = os.path.join(path_tex, "blocks_interactions", "prebuilt_images")
seed = 112358

# use CUDA if available
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
used_cuda = "used" if use_cuda else "not used"


def power(X, maxiter):
    return power_method(X, "X", eps=1e-4, maxiter=maxiter)


def linalg(X, **kwargs):
    return torch.linalg.norm(X, 2) ** 2


def lanczos_scipy(X):
    return svds(X.cpu().numpy(), k=1)[1][0] ** 2


def lanczos_tch(X, maxiter):
    return Lanczos(X, which="X", n_cv=maxiter)


def prepare(X, many):
    # reproducibility
    np.random.seed(11235813)
    torch.manual_seed(11235813)

    X_ = X[:, :many]
    n, p = X_.shape
    X_mean, X_std, _, _ = cpt_mean_std(X_)
    X_normalized = (X_ - X_mean) / X_std
    X_tch = torch.from_numpy(X_normalized).float().to(device)
    ref = torch.linalg.norm(X_tch, 2) ** 2
    print("-------------Finished computing means and sd and to gpu")
    return ref, X_tch.contiguous()


def run_routine(X_, routine, maxiter):
    t_0 = time.perf_counter()
    val = routine(X_, maxiter=maxiter)
    if use_cuda:
        torch.cuda.synchronize()
    time_ = time.perf_counter() - t_0
    return val, time_


def benchmark(X, sizes, dic_args, n_repeat=1):
    time_plot = []
    val_plot = []
    ref_plot = []
    for idx, size in enumerate(sizes):
        print("\n", f"\r ##### Begin size = {size:.3f}, {idx+1}/{len(sizes)}.")
        times_size = []
        ref, X_ = prepare(X, 500)
        ref_plot.append(ref)

        for n in range(n_repeat + 1):
            val, time_ = run_routine(X_, dic_args["routine"], maxiter=size)
            if n != 0:
                times_size.append(time_)
                print(f"Finished {n} out of {n_repeat}", f"time = {time_:.3f}s." "")
            else:
                val_plot.append(val.cpu().numpy())
                print(f"""Finished {n} (warmup) out of {n_repeat}.""")
        if n_repeat > 0:
            time_plot.append(np.median(times_size))
    return val_plot, time_plot


def match_routine_args(routine_name):
    dict_args = {
        "Torch": {
            "routine": linalg,
        },
        "PowerMethod": {
            "routine": power,
        },
        "scipy": {
            "routine": lanczos_scipy,
        },
        "lanczos": {
            "routine": lanczos_tch,
        },
    }
    return dict_args[routine_name]


def run_full_benchmark(routines_names, sizes, n_repeat=20):
    dict_times = {name: [] for name in routines_names}
    dict_vals = {name: [] for name in routines_names}

    for idx, routine in enumerate(routines_names):
        print(
            f"\033[1;%dm #########" % 34 + "Begin " + routine + "#########\033[0m"
        )  # noqa
        dict_args = match_routine_args(routine)
        n_repeat = n_repeat if idx != 0 else 1
        vals, times = benchmark(X, sizes, dict_args, n_repeat=n_repeat)
        dict_times[routine].append(times)
        dict_vals[routine].append(vals)
    return dict_times, dict_vals


if __name__ == "__main__":

    #############################
    # Load the data
    #############################

    print(f"Cuda is {used_cuda}.")

    # get the data
    n_samples, n_features = 20000, 500
    X = np.random.randn(20000, 500)

    print("~~~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~~~")

    #######################
    # Benchmark
    #######################

    iterates = np.arange(1, 50, 4)
    routines_names = ["Torch", "PowerMethod", "lanczos"]

    times_to_plot, vals = run_full_benchmark(routines_names, iterates, n_repeat=10)
    markers = ["+", "o", "^", "|", "*"]

    plt.figure()
    for idx, name in enumerate(routines_names):
        if idx == 0:
            plt.plot(
                iterates, times_to_plot[name][0], "--", marker=markers[idx], label=name
            )
        else:
            plt.plot(iterates, times_to_plot[name][0], marker=markers[idx], label=name)
    plt.xlabel("Iterate number")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(path_save, f"benchmark_powermethod_scipy.pdf"))

    plt.figure()
    for idx, name in enumerate(routines_names[1:]):
        plt.plot(
            iterates,
            np.array(vals[name][0]) - np.array(vals["Torch"][0]),
            marker=markers[idx],
            label=name,
        )
    plt.xlabel("Iterate number")
    plt.ylabel("Difference against PyTorch 2-norm")
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(path_save, f"benchmark_powermethod_scipy_values.pdf"))
    plt.show()
