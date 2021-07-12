
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from interactionsmodel import path_data, path_tex

plt.rcParams.update({"font.size": 16})
sns.set()
path_save = os.path.join(path_tex, "blocks_interactions", "prebuilt_images")

n = 500
x = np.linspace(-2, 2, n)
y = np.abs(x)


plt.figure()
plt.plot(x, y, color="r", lw=2.5, label=r"$|x|$")
need_label = True
for fact in [-.9, -.75, -.3, 0, .3, .75, .9]:
    if need_label:
        plt.plot(x, fact*x, "--", color="b", label='subgradients at origin')
        need_label = False
    else:
        plt.plot(x, fact*x, "--", color="b")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim([-1, 2])
plt.legend(loc="upper center")
plt.tight_layout()
plt.show()