import numpy as np

from boa.models.multi_output_gp_regression_model import MultiOutputGPRegressionModel
from .abstract import AbstractAcquisition
from .util import AcquisitionError


class ExpectedImprovement(AbstractAcquisition):

    def __init__(self, xi):

        super().__init__()

        self.xi = xi

    def evaluate(self, model: MultiOutputGPRegressionModel, xs: np.ndarray, ys: np.ndarray, candidate_xs: np.ndarray) -> np.ndarray:

        if len(ys.shape) < 2:
            ys = ys.reshape([-1, 1])
        elif len(ys.shape) == 2 and ys.shape[1] > 1:
            raise AcquisitionError("ys must be one dimensional!")
        elif len(ys.shape) > 2:
            raise AcquisitionError("ys must be at most rank 2!")

        # Get candidate mean and variance predictions
        means, var = model.predict(candidate_xs, numpy=True)

        # Find the argmax from the candidates
