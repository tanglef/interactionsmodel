import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from interactionsmodel import path_tex
import os

plt.rcParams.update({"font.size": 14})
sns.set()

transcription_factor = True
if transcription_factor is False:
    n_features = np.arange(10, 160 + 1)
else:
    n_features = np.arange(10, 560 + 1)

n_squarefeatures = n_features * (n_features + 1) / 2
n_interactions = n_features + n_squarefeatures
# n__pure_squarefeatures = n_features*(n_features-1)/2


plt.figure(figsize=(14, 7))
plt.plot(n_features, n_features, label="model without. interaction")
plt.plot(n_features, n_interactions, label="model with interaction")
plt.yscale("log")
plt.xlabel("Number of main features")
plt.ylabel("Number of features in the model")
plt.title("Evolution of the number of features with/without interaction")
plt.legend()
# plt.savefig('./save_fig/ev_nb_features.pdf')

plt.figure(figsize=(14, 7))
n_samples_list = np.array([5000, 10000, 15000, 20000])
size_float64 = np.ones(1).nbytes
for i in n_samples_list:
    mem = i * n_interactions * size_float64
    mem /= 10 ** 9
    plt.plot(n_features, mem, label="Nb of samples : " + str(i))
plt.yscale("log")
plt.xlabel("Number of main features")
plt.ylabel("Size (Gb) of the desgin matrix with interaction")
plt.title("Evolution of the number of features with/without interaction")
plt.legend()


# sns.set_context('poster')
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(121)
ax1.plot(n_features, n_features, label="model without interaction")
ax1.plot(n_features, n_interactions, label="model with interaction")
ax1.set_yscale("log")
ax1.set_xlabel("Number of main features p")
ax1.set_ylabel("Number of features in the model")
ax1.legend()  # using a named size

ax2 = plt.subplot(122)
n_samples_list = np.array([5000, 10000, 15000, 20000])
size_float64 = np.ones(1).nbytes
for i in n_samples_list:
    mem = i * n_interactions * size_float64
    mem /= 10 ** 9
    ax2.plot(n_features, mem, label="Nb of samples : " + str(i))
ax2.set_yscale("log")
ax2.set_xlabel("Number of main features p")
ax2.set_ylabel("Size (Gb)")
ax2.legend
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(path_tex, "size_matrix.pdf"))
