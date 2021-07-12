"""
================================================
Try to explore why condition number is so high
================================================
"""

import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
from interactionsmodel import path_data, path_tex
from interactionsmodel.utils import cpt_mean_std, whitening
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
MAX_ITER = int(500)

if __name__ == "__main__":
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
    meanX, stdX, meanZ, stdZ = cpt_mean_std(X, full=False)
    X_ = X.copy()
    X -= meanX
    X /= stdX

    # condX = np.linalg.cond(X)

    # cond_bb, cond_bbws = [], []
    # for i in range(X.shape[1]):
    #     print("Block", i)
    #     Zi = (X_[:, i].reshape(-1, 1) * X_[:, i:] - meanZ[i]) / stdZ[i]
    #     cond_bb.append(np.linalg.cond(Zi))

    # plt.figure()
    # plt.plot(np.arange(len(cond_bb)), cond_bb, label="")
    # plt.xlabel("Block number")
    # plt.ylabel("Condition number")
    # plt.legend()
    # plt.yscale("log")
    # plt.savefig(os.path.join(path_save, "condition_number_genom.pdf"))
    # plt.show()

    # plt.figure()
    # ax = sns.heatmap(X[:, :60].T @ X[:, :60] / X.shape[0],
    #                  xticklabels=False, yticklabels=False, square=True)
    # ax.set_aspect("equal")
    # plt.savefig(os.path.join(path_save, "Gram_matrix_genom_60.pdf"))

    # plt.figure()
    # ax = sns.heatmap(X.T @ X / X.shape[0],
    #                  xticklabels=False, yticklabels=False, square=True)
    # ax.set_aspect("equal")
    # plt.savefig(os.path.join(path_save, "Gram_matrix_genom.pdf"))

    # plt.show()

    data = X_[:, 80:90]
    import pandas as pd

    data = pd.DataFrame(data)
    ax = sns.boxenplot(data=data, k_depth=6, showfliers=False)
    plt.savefig(os.path.join(path_save, "boxenplot_genom.pdf"))
