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
    tmp = X.T @ resid
    return np.max(np.abs(1 / n * tmp + reg * w))


seed = 11235813
np.random.seed(seed)

# The desired mean values of the sample.
# n, p = 1000, 1000
# mu = np.zeros(p)
# # The desired covariance matrix.
# r = np.ones(p) * np.random.rand(p, p)
# # Generate the random samples.
# X = np.random.multivariate_normal(mu, r, size=n)
# X, _ = make_regression(n, p, random_state=seed)
# X = .9 * np.ones(n) + .1 * np.eye(n)
# sparsity = p
maxiter = int(1e4)
# snr = 5

# wstar = np.zeros(p)
# choice_array = np.arange(p)
# w_indx = np.random.choice(choice_array,
#                           size=sparsity,
#                           replace=False)
# w_value = np.random.choice([-1, 1], size=sparsity)
# wstar[w_indx] = w_value
# y = X @ wstar
# vary = np.var(y, ddof=0)
# vareps = vary / snr
# y += np.sqrt(vareps) * np.random.rand(n)

from sklearn.datasets import fetch_openml

dataset = fetch_openml("leukemia")
reg = 0.1
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
        grad = 1 / n * X.T @ (X @ w - y) + reg * w
        w -= 1 / Ls[iL] * grad
        dists[iL].append(d_primal(X, y, w, reg))
        if dists[iL][-1] < 1e-12:
            cvs[iL] = idx + 1
            break

path_save = os.path.join(path_tex, "blocks_interactions", "prebuilt_images")
plt.figure()
for idx, lab in enumerate(Ls):
    plt.plot(np.arange(cvs[idx] + 1), dists[idx], label=f"1/({divs[idx]:.0f}L)")
plt.xlabel(r"epoch $k$")
plt.ylabel(r"$d(0, \mathcal{P}(\beta^{k}))$")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(path_save, "importance_step_size.pdf"))
plt.show()
