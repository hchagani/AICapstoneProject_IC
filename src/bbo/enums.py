from enum import Enum

from sklearn.gaussian_process.kernels import Matern, RBF


class KernelType(Enum):
    RBF = RBF
    MATERN = Matern
