import numpy as np
import os
from sklearn.datasets import make_regression
import seaborn as sns
import matplotlib.pyplot as plt
from interactionsmodel import path_tex

sns.set()
plt.rcParams.update({"font.size": 16})


def d_primal(X, y, w, l1, l2):
    n = X.shape[0]
    resid = y - X @ w
    tmp = X.T @ resid / n - w * l2
    tmp -= np.clip(tmp, -l1, l1)
    return np.max(np.abs(tmp))


def prox(w, mu, l1, l2):
    tmp = w - np.clip(w, -mu*l1, mu*l1)
    return tmp / (1 + mu*l2)



seed = 11235813
np.random.seed(seed)
maxiter = int(1e4)

from sklearn.datasets import fetch_openml

dataset = fetch_openml("leukemia")
l1, l2 = 0.5, 0.5
y = 2 * ((dataset.target != "AML") - 0.5)
X = np.asfortranarray(dataset.data.astype(float))
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
n, p = X.shape
y -= np.mean(y)
y /= np.std(y)
print(f"lambda max = {np.abs(np.max(X.T @ y)) / n:.3f}")
divs = [1.0, 2.0, 5.0, 10.0, 100.0]
dists = [[] for _ in range(5)]

L = np.linalg.norm(X, 2) ** 2 / n
L2 = [.2, .5, 1, 2., 5.]
cvs = [maxiter] * 5
for il2, l2 in enumerate(L2):
    w = np.zeros(p)
    dists[il2].append(d_primal(X, y, w, l1, l2))
    for idx in range(maxiter):
        grad = 1 / n * X.T @ (X @ w - y)
        w -= 1 / L * grad
        w = prox(w, 1 / L, l1, l2)
        dists[il2].append(d_primal(X, y, w, l1, l2))
        if dists[il2][-1] < 1e-12:
            cvs[il2] = idx + 1
            break

path_save = os.path.join(path_tex, "blocks_interactions", "prebuilt_images")
plt.figure()
for idx, lab in enumerate(L2):
    plt.plot(np.arange(cvs[idx] + 1), dists[idx], label=f"l2={L2[idx]:.1f}")
plt.xlabel(r"epoch $k$")
plt.ylabel(r"$d(0, \partial\mathcal{P}(\beta^{k}))$")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(path_save, "cv_rate_enet.pdf"))
plt.show()
