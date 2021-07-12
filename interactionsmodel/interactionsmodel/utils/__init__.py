from .products import product_Z_full, product_ZT_full, product_ZT, product_Z

PRODS = {"full": (product_Z_full, product_ZT_full), "small": (product_Z, product_ZT)}

from .whitening_zca import whitening  # noqa:E402 F401
from .pgd_utils import power_method, Lanczos  # noqa:E402 F401
from .pre_process import preprocess  # noqa:E402 F401
import interactionsmodel.utils.numba_fcts  # noqa:E402 F401
from .build_interactions import make_Z_full, make_Z  # noqa:E402 F401
from .compat_helper import *  # noqa:E402 F401
from .make_dataset import (
    make_data,
    get_lambda_max,  # noqa:E402 F401
    make_regression,
)  # noqa:E402 F401
from .kkt_violation import (
    kkt_violation,
    kkt_nb,
    kkt_violation_tch,
)  # noqa:E402 F401 E501
