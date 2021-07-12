import numpy as np
import os
from sklearn.datasets import make_regression
import seaborn as sns
import matplotlib.pyplot as plt
from interactionsmodel import path_tex

sns.set()
plt.rcParams.update({"font.size": 16})


def d_primal(X, y, w, reg):
    n = X.shape[0]
    resid = X @ w - y
    tmp = X.T @ resid / n
    tmp -= np.clip(tmp, -reg, reg)
    return np.max(np.abs(tmp))


def prox(w, reg):
    return w - np.clip(w, -reg, reg)


seed = 11235813
np.random.seed(seed)
maxiter = int(1e4)

from sklearn.datasets import fetch_openml

dataset = fetch_openml("leukemia")
reg = 0.3
y = 2 * ((dataset.target != "AML") - 0.5)
X = np.asfortranarray(dataset.data.astype(float))
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
n_samples = len(y)
n, p = X.shape
y -= np.mean(y)
y /= np.std(y)
divs = [1.0, 2.0, 5.0, 10.0, 100.0]
w = np.zeros(p)
dists = [[d_primal(X, y, w, reg)] for _ in range(5)]

L = np.linalg.norm(X, 2) ** 2 / n
Ls = [L * i for i in divs]
cvs = [maxiter] * 5
for iL, L in enumerate(Ls):
    w = np.zeros(p)
    for idx in range(maxiter):
        grad = 1 / n * X.T @ (X @ w - y)
        w -= 1 / Ls[iL] * grad
        w = prox(w, reg / Ls[iL])
        dists[iL].append(d_primal(X, y, w, reg))
        if dists[iL][-1] < 1e-12:
            cvs[iL] = idx + 1
            break

path_save = os.path.join(path_tex, "blocks_interactions", "prebuilt_images")
plt.figure()
for idx, lab in enumerate(Ls):
    plt.plot(np.arange(cvs[idx] + 1), dists[idx], label=f"1/({divs[idx]:.0f}L)")
plt.xlabel(r"epoch $k$")
plt.ylabel(r"$d(0, \partial\mathcal{P}(\beta^{k}))$")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(path_save, "cv_rate_lasso.pdf"))
plt.show()
