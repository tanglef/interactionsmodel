"""
====================================
Some descriptions of the data used
====================================
"""

import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import interactionsmodel  # noqa for setting numba/numpy threads
from interactionsmodel.solvers import CD
from interactionsmodel.solvers import CBPG_CS as CBPG
from interactionsmodel import path_data, path_tex
from interactionsmodel.utils import make_regression, kkt_violation
from interactionsmodel.utils import get_lambda_max
import time
from interactionsmodel.data_management import download_data
from sklearn.model_selection import train_test_split

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

data = pd.DataFrame({"gene expression": y})
# for y

sns.kdeplot(
    data=data,
    x="gene expression",
    fill=True,
    palette="crest",
)
plt.savefig(os.path.join(path_save, "distrib_gene_expression.pdf"))
