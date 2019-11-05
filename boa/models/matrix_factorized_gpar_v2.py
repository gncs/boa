from typing import Tuple, List

import numpy as np
import tensorflow as tf
from varz.tensorflow import Vars, minimise_l_bfgs_b

from .abstract_model_v2 import ModelError
from .gpar_v2 import GPARModel


class MatrixFactorizedGPARModel(GPARModel):

    def __init__(self,
                 kernel: str,
                 num_optimizer_restarts: int,
                 verbose: bool = False,
                 name="matrix_factorized_gpar_model",
                 **kwargs):

        super(MatrixFactorizedGPARModel, self).__init__(kernel=kernel,
                                                        num_optimizer_restarts=num_optimizer_restarts,
                                                        verbose=verbose,
                                                        name=name,
                                                        **kwargs)
