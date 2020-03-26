import abc

import numpy as np

from boa.models.multi_output_gp_regression_model import MultiOutputGPRegressionModel


class AbstractAcquisition:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def evaluate(self, model: MultiOutputGPRegressionModel, xs: np.ndarray, ys: np.ndarray, candidate_xs: np.ndarray) -> np.ndarray:
        pass
