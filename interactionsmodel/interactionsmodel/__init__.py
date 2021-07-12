__version__ = "0.0.0"

import interactionsmodel.solvers
import interactionsmodel.data_management
import interactionsmodel.utils
import os
import sys
import numba

numba.set_num_threads(8)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

path_file = os.path.dirname(__file__)
path_tex = os.path.join(path_file, "..", "TeX")
path_data = os.path.join(path_file, "data_management")

sys.path.insert(0, os.path.join(path_file, "..", "benchmarks"))
import benchmarks.bench_products
