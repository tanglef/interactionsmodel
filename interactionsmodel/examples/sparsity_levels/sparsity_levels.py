"""
====================================
Look where the coordinates a block
====================================
"""

import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
from interactionsmodel.solvers import PGD, CD, CBPG_CS, CBPG_CS_mod, CBPG_permut
from interactionsmodel import path_data, path_tex
from interactionsmodel.utils import make_regression
from interactionsmodel.utils import get_lambda_max
import json


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
EPS = 1e-5
MAX_ITER = int(300)


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


def bench_CD(instance, sp_beta, sp_theta):
    instance.run(MAXITER, eps=EPS)
    theta_i = instance.theta
    beta_i = instance.beta
    sp_beta.append(np.linalg.norm(beta_i, 0))
    sp_theta.append(np.linalg.norm(theta_i, 0))


def bench_CBPG(instance, sp_beta, sp_theta):
    instance.run(MAXITER, eps=EPS)
    theta_i = instance.theta
    beta_i = instance.beta
    sp_beta.append(torch.linalg.norm(beta_i.flatten(), 0).cpu().numpy().item())
    sp_theta.append(torch.linalg.norm(theta_i.flatten(), 0).cpu().numpy().item())


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
    sp_beta, sp_theta, callback = get_callback(instance)
    if routine == cd:
        bench_CD(instance, sp_beta, sp_theta)
    else:
        bench_CBPG(instance, sp_beta, sp_theta)
    return sp_beta, sp_theta


def benchmark(X, y, l_lambdas, dic_args, datatype):
    l1 = []
    l2 = []
    spb = []
    spt = []
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

        sp_beta, sp_theta = run_routine(
            X_,
            y_,
            lambdas,
            device_,
            full,
            dic_args["routine"],
            acceleration=acceleration,
        )
        if device_ == "cpu":
            l1.append(lambdas[0])
            l2.append(lambdas[2])
            spb.append(sp_beta)
            spt.append(sp_theta)
        else:
            l1.append(lambdas[0].cpu().numpy().item())
            l2.append(lambdas[2].cpu().numpy().item())
            spb.append(sp_beta)
            spt.append(sp_theta)
    return l1, l2, spb, spt


def run_full_benchmark(routines_names, l_lambdas, datatype="double"):
    dict_ = {
        name: {"lambda_1": [], "lambda_2": [], "sp_beta": [], "sp_theta": []}
        for name in routines_names
    }
    for routine in routines_names:
        print(
            f"\033[1;%dm #########" % 34 + "Begin " + routine + "#########\033[0m"
        )  # noqa
        print("~~~~~~~~ Beginning warmup")
        global MAXITER
        MAXITER = 1  # warmup

        dict_args = match_routine_args(routine)
        _ = benchmark(X, y, [l_lambdas[0]], dict_args, datatype)
        print("~~~~~~~~ Beginning benchmark")
        MAXITER = MAX_ITER
        dict_args = match_routine_args(routine)
        l1, l2, spb, spt = benchmark(X, y, l_lambdas, dict_args, datatype)
        dict_[routine]["lambda_1"].append(l1)
        dict_[routine]["lambda_2"].append(l2)
        dict_[routine]["sp_beta"].append(spb)
        dict_[routine]["sp_theta"].append(spt)
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
    sp_beta = []
    sp_theta = []

    def callback(it, beta_i, theta_i):
        reached_max = it != MAXITER
        return reached_max

    return sp_beta, sp_theta, callback


if __name__ == "__main__":
    save_fig = True

    #############################
    # Load the data
    #############################

    print(f"Cuda is {used_cuda}.")
    snr = 10
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
    )
    X_ = X.copy()
    n_lambda = 10
    lambda_1, lambda_2 = 200, 200
    l_alpha = [
        (lb1, lb1, lb2, lb2)
        for lb1 in [lambda_1 / i for i in np.logspace(-0.2, 3.5, num=n_lambda)]
        for lb2 in [lambda_2 / i for i in np.logspace(-0.2, 3.5, num=n_lambda)]
    ]

    print("~~~~~~~~~~~~~~~~ Finished preparing data ~~~~~~~~~~~~~~~~~~")

    routines_names = ["CD"]
    dic_to_plot = run_full_benchmark(
        routines_names, l_lambdas=l_alpha, datatype="float"  # double to have 64
    )
    dic_to_plot["lambda_max"] = [lambda_1, lambda_2]
    print("plotting")
    for name in routines_names:
        dic_to_plot[name]["lambda_1"] = dic_to_plot[name]["lambda_1"][0]
        dic_to_plot[name]["lambda_2"] = dic_to_plot[name]["lambda_2"][0]
        dic_to_plot[name]["sp_beta"] = dic_to_plot[name]["sp_beta"][0]
        dic_to_plot[name]["sp_theta"] = dic_to_plot[name]["sp_theta"][0]
        dic_to_plot[name]["sp_beta"] = [
            val[0] / n_features for val in dic_to_plot[name]["sp_beta"]
        ]
        dic_to_plot[name]["sp_theta"] = [
            val[0] / n_inter for val in dic_to_plot[name]["sp_theta"]
        ]

    class Serializer(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            return json.JSONEncoder.default(self, obj)

    with open(os.path.join(path_data, "sp_levels.json"), "w") as json_file:
        json.dump(dic_to_plot, json_file, cls=Serializer, indent=4)
