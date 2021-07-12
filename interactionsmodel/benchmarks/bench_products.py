import numpy as np
import torch
import os
import seaborn as sns
import time
import matplotlib.pyplot as plt
import warnings
from numba import njit
from interactionsmodel.data_management import download_data
from interactionsmodel.utils.numba_fcts import cpt_mean_std
from interactionsmodel import path_tex, path_data

try:
    from pykeops.torch import Vi, Vj, Genred
except ImportError:
    warnings.warn("pykeops.torch could not be loaded. Check install.")


plt.rcParams.update({"font.size": 16})
save_fig = False
pre_compile = True
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
path_target = os.path.join(
    path_data, "Data", "genes_data_predicted_and_predictive_variables"
)
path_save = os.path.join(path_tex, "blocks_interactions", "prebuilt_images")
seed = 112358
# use CUDA if available
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.float32
used_cuda = "used" if use_cuda else "not used"


##########################################
# Functions product Z @ beta to benchmark
##########################################


def product_Z_keops(X, XT, beta, res):
    """Make the product Z @ beta by blocks w/ KeOps."""
    _, p = X.shape
    obegin = 0
    for var in range(p):
        p_tilde = p - var
        x_ = Vi(XT[var, :].view(-1, 1))  # n, 1
        X_i = Vi(X[:, var:].contiguous())  # n, 1, p-var
        Z = x_ * X_i  # n, 1, p-var
        b_tilde = Vj(
            beta[
                obegin : (obegin + p_tilde),
            ].view(1, -1)
        )
        K2 = Z.matvecmult(b_tilde)  # Lz
        res += K2.sum(1)
        obegin += p_tilde
    return res


def product_Z_torch(X, XT, beta, res):
    """Make the product Z @ beta by blocks w/ torch."""
    _, p = X.shape
    obegin = 0
    for var in range(p):
        p_tilde = p - var
        Xi = X[:, var].view(-1, 1)
        K = Xi * X[:, var:]
        res += (
            K
            @ beta[
                obegin : (obegin + p_tilde),
            ].view(-1, 1)
        )
        obegin += p_tilde
    return res


def product_Z_full(X, XT, beta, res):
    _, p = X.shape
    formula = f"MatVecMult((Var(0,1,0) * Var(1,{p},0)), Var(2,{p},1))"
    variables = ["Var(0,1,0)", f"Var(1,{p},0)", f"Var(2,{p},1)"]
    my_routine = Genred(formula, variables, reduction_op="Sum", axis=1, dtype="float32")
    obegin = 0
    for var in range(p):
        x_ = XT[var, :].view(-1, 1)
        b_tilde = beta[
            obegin : (obegin + p),
        ].view(1, -1)
        res += my_routine(x_, X, b_tilde)
        obegin += p
    return res


@njit()
def product_Z_cd_enet(X, XT, beta, res):
    _, p = X.shape
    jj = 0
    for j1 in range(p):
        for j2 in range(j1, p):
            Z_tmp = X[:, j2] * X[:, j1]
            res += Z_tmp * beta[jj]
            jj += 1
    return res


#############################################
# Functions product Z.T @ theta to benchmark
#############################################


def product_ZT_torch(X, XT, theta, res):
    """Make the product Z.T @ beta by blocks w/ torch."""
    n, p = X.shape
    n_breaks = 20
    breaks = torch.linspace(0, n, n_breaks + 1, dtype=torch.int)
    obegin = 0
    theta = theta.view(1, -1)
    for var in range(X.shape[1]):
        p_tilde = p - var
        for k, nk in enumerate(breaks[:-1]):
            nk1 = breaks[k + 1]
            xi = X[nk:nk1, var].view(-1, 1)
            Z = xi * X[nk:nk1, var:]
            res[obegin : (obegin + p_tilde)] += Z.T @ theta[:, nk:nk1].view(-1, 1)
        obegin += p_tilde
    return res


def product_ZT_keops(X, XT, theta, res):
    """Make the product Z.T @ beta by blocks w/ KeOps."""
    n, p = X.shape
    n_breaks = 20
    breaks = torch.linspace(0, n, n_breaks + 1, dtype=torch.int)
    obegin = 0
    theta = theta.view(1, -1)
    break_l1 = n // n_breaks
    break_l2 = break_l1 + 1 if n != n_breaks * break_l1 else break_l1
    formula_1 = f"MatVecMult((Var(0,{break_l1},1) * Var(1,{break_l1},0)), Var(2,{break_l1},1))"  # noqa
    variables_1 = [
        f"Var(0,{break_l1},1)",
        f"Var(1,{break_l1},0)",
        f"Var(2,{break_l1},1)",
    ]
    my_routine_1 = Genred(
        formula_1, variables_1, reduction_op="Sum", axis=1, dtype="float32"
    )
    formula_2 = f"MatVecMult((Var(0,{break_l2},1) * Var(1,{break_l2},0)), Var(2,{break_l2},1))"  # noqa
    variables_2 = [
        f"Var(0,{break_l2},1)",
        f"Var(1,{break_l2},0)",
        f"Var(2,{break_l2},1)",
    ]
    my_routine_2 = Genred(
        formula_2, variables_2, reduction_op="Sum", axis=1, dtype="float32"
    )
    for var in range(p):
        p_tilde = p - var
        for k, nk in enumerate(breaks[:-1]):
            nk1 = breaks[k + 1]
            x_ = XT[var, nk:nk1].view(1, -1)  # 1, 1, n
            X_i = XT[var:, nk:nk1].contiguous()  # 1, p-var, n
            if nk1 - nk == break_l1:
                res[
                    obegin : (obegin + p_tilde),
                ] += my_routine_1(x_, X_i, theta[:, nk:nk1])
            else:
                res[
                    obegin : (obegin + p_tilde),
                ] += my_routine_2(x_, X_i, theta[:, nk:nk1])
        obegin += p_tilde
    return res


def product_ZT_full(X, XT, theta, res):
    n, p = X.shape
    n_breaks = 20
    breaks = torch.linspace(0, n, n_breaks + 1, dtype=torch.int)
    obegin = 0
    break_l1 = n // n_breaks
    break_l2 = break_l1 + 1 if n != n_breaks * break_l1 else break_l1
    formula_1 = f"MatVecMult((Var(0,{break_l1},1) * Var(1,{break_l1},0)), Var(2,{break_l1},1))"  # noqa
    variables_1 = [
        f"Var(0,{break_l1},1)",
        f"Var(1,{break_l1},0)",
        f"Var(2,{break_l1},1)",
    ]
    my_routine_1 = Genred(
        formula_1, variables_1, reduction_op="Sum", axis=1, dtype="float32"
    )
    formula_2 = f"MatVecMult((Var(0,{break_l2},1) * Var(1,{break_l2},0)), Var(2,{break_l2},1))"  # noqa
    variables_2 = [
        f"Var(0,{break_l2},1)",
        f"Var(1,{break_l2},0)",
        f"Var(2,{break_l2},1)",
    ]
    my_routine_2 = Genred(
        formula_2, variables_2, reduction_op="Sum", axis=1, dtype="float32"
    )
    theta = theta.view(1, -1)
    for var in range(p):
        for k, nk in enumerate(breaks[:-1]):
            nk1 = breaks[k + 1]
            x_ = XT[var, nk:nk1].view(1, -1)
            X_i = XT[:, nk:nk1].contiguous()
            if nk1 - nk == break_l1:
                res[
                    obegin : (obegin + p),
                ] += my_routine_1(x_, X_i, theta[:, nk:nk1])
            else:
                res[
                    obegin : (obegin + p),
                ] += my_routine_2(x_, X_i, theta[:, nk:nk1])
        obegin += p
    return res


@njit()
def product_ZT_cd_enet(X, XT, theta, res):
    _, p = X.shape
    jj = 0
    for j1 in range(p):
        for j2 in range(j1, p):
            Z_tmp = X[:, j2] * X[:, j1]
            res[jj] += np.dot(Z_tmp.T, theta)[0]
            jj += 1
    return res


##############################
# Perform the Benchmark
##############################


def prepare(X, many, transpose=False, use_torch=False, dtype=None, full=False):
    # reproducibility
    np.random.seed(11235813)
    torch.manual_seed(11235813)

    X_ = X[:, :many]
    n, p = X_.shape
    X_mean, X_std, _, _ = cpt_mean_std(X_)
    X_normalized = (X_ - X_mean) / X_std
    q = int(p * (p + 1) / 2) if not full else int(p ** 2)
    vect = np.random.randn(q) if not transpose else np.random.randn(n)
    vect = vect.reshape(-1, 1)
    res = np.zeros((q)) if transpose else np.zeros((n))
    print("-------------Finished computing means and sd")
    if use_torch:
        X_tch = torch.from_numpy(X_normalized).type(dtype)
        vect = torch.from_numpy(vect).type(dtype)
        res = torch.from_numpy(res).view(-1, 1).type(dtype)
        return X_tch.contiguous(), X_tch.T.contiguous(), vect, res
    else:
        X_normalized = X_normalized.astype("float32")
        vect = vect.astype("float32")
        res = res.astype("float32")
        return X_normalized, X_normalized.T, vect, res


def run_routine(X_, XT_, vect, res, routine):
    t_0 = time.perf_counter()
    routine(X_, XT_, vect, res)
    if use_cuda:
        torch.cuda.synchronize()
    time_ = time.perf_counter() - t_0
    return time_


def benchmark(X, sizes, transpose, dic_args, n_repeat=20):
    time_plot = []
    for idx, size in enumerate(sizes):
        print("\n", f"\r ##### Begin size = {size:.3f}, {idx+1}/{len(sizes)}.")
        times_size = []
        X_, XT_, vect, res = prepare(
            X,
            size,
            use_torch=dic_args["use_torch"],
            dtype=dic_args["dtype"],
            transpose=transpose,
            full=dic_args["full"],
        )

        for n in range(n_repeat + 1):
            time_ = run_routine(X_, XT_, vect, res, dic_args["routine"])
            if n != 0:
                times_size.append(time_)
                print(
                    f"Finished {n} out of {n_repeat}", f"time Z @ beta={time_:.3f}s." ""
                )
            else:
                print(f"""Finished {n} (warmup) out of {n_repeat}.""")
        if n_repeat > 0:
            time_plot.append(times_size)
    return time_plot


def match_routine_args(routine_name, trans=False):
    dict_args = {
        "KeOps": {
            "routine": product_ZT_keops if trans else product_Z_keops,
            "use_torch": True,
            "dtype": torch.cuda.FloatTensor,
            "full": False,
        },
        "Torch": {
            "routine": product_ZT_torch if trans else product_Z_torch,
            "use_torch": True,
            "dtype": torch.FloatTensor,
            "full": False,
        },
        "Torch+CUDA": {
            "routine": product_ZT_torch if trans else product_Z_torch,
            "use_torch": True,
            "dtype": torch.cuda.FloatTensor,
            "full": False,
        },
        "KeOps+full": {
            "routine": product_ZT_full if trans else product_Z_full,
            "use_torch": True,
            "dtype": torch.cuda.FloatTensor,
            "full": True,
        },
        "Numba": {
            "routine": product_ZT_cd_enet if trans else product_Z_cd_enet,
            "use_torch": False,
            "dtype": "float32",
            "full": False,
        },
    }
    return dict_args[routine_name]


def run_full_benchmark(routines_names, sizes, n_repeat=20):
    dict_times = {name: [] for name in routines_names}
    for trans in [True, False]:
        for routine in routines_names:
            print(
                f"\033[1;%dm #########" % 34 + "Begin " + routine + "#########\033[0m"
            )  # noqa
            print(f"\r With tranpose={trans}")
            dict_args = match_routine_args(routine, trans)
            times = benchmark(X, sizes, trans, dict_args, n_repeat=n_repeat)
            dict_times[routine].append(times)
    return dict_times


if __name__ == "__main__":

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
    y = y[:, 0]

    print("~~~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~~~")

    #######################
    # Benchmark
    #######################

    sizes_X = [65, 150, 200, 250, 300, 400, 500, 531]
    routines_names = ["Torch", "Numba", "Torch+CUDA"]
    if pre_compile:  # make reading easier in log
        run_full_benchmark(["Numba"], [sizes_X[-1]], n_repeat=0)

    times_to_plot = run_full_benchmark(routines_names, sizes_X, n_repeat=20)
    import pandas as pd
    df = pd.DataFrame(pd.DataFrame(times_to_plot).iloc[[0]])
    df2 = df.apply(lambda x: [j for i in x for j in i]).apply(
        lambda x: [j for i in x for j in i])
    df2 = df2.reset_index().melt(id_vars='index')
    df2["sizes"] = np.hstack(np.tile(np.repeat(sizes_X, 20), (3, 1)))
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df2,
        x="sizes", y="value", hue="variable",
        markers=False, dashes=False, err_style="band",
        style="variable"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:])
    ax.set(xlabel='Number of variables in X', ylabel='Time (s)')
    fig.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(path_save, "benchmark_prod_ZTtheta.pdf"))

    df = pd.DataFrame(pd.DataFrame(times_to_plot).iloc[[1]])
    df2 = df.apply(lambda x: [j for i in x for j in i]).apply(
        lambda x: [j for i in x for j in i])
    df2 = df2.reset_index().melt(id_vars='index')
    df2["sizes"] = np.hstack(np.tile(np.repeat(sizes_X, 20), (3, 1)))
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df2,
        x="sizes", y="value", hue="variable",
        markers=False, dashes=False, err_style="band",
        style="variable"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:])
    ax.set(xlabel='Number of variables in X', ylabel='Time (s)')
    fig.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(path_save, "benchmark_prod_Zbeta.pdf"))

    plt.show()
    # markers = ["+", "o", "^", "|", "*"]

    # # Z @ beta
    # plt.figure()
    # for idx, name in enumerate(routines_names):
    #     plt.plot(sizes_X, times_to_plot[name][1], marker=markers[idx], label=name)
    # plt.xlabel("Number of variables in X")
    # plt.ylabel("Time (s)")
    # plt.legend()
    # plt.tight_layout()

    # if save_fig:
    #     plt.savefig(os.path.join(path_save, "benchmark_prod_Zbeta.pdf"))

    # plt.figure()
    # for idx, name in enumerate(routines_names):
    #     plt.plot(sizes_X, times_to_plot[name][0], marker=markers[idx], label=name)
    # plt.xlabel("Number of variables in X")
    # plt.ylabel("Time (s)")
    # plt.legend()
    # plt.tight_layout()

    # if save_fig:
    #     plt.savefig(os.path.join(path_save, "benchmark_prod_ZTtheta.pdf"))

    # plt.show()
