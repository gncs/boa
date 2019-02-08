import numpy as np

from boa.models.abstract import AbstractModel
from .abstract import AbstractAcquisition
from .util import calculate_hypervolume, normalize, get_frontier


class SMSEGO(AbstractAcquisition):
    NOISE = 1E-10

    def __init__(self, gain: float, epsilon: float, reference_point: np.ndarray, mean: float, std: float) -> None:
        super().__init__()

        self.gain = gain
        self.epsilon = epsilon
        self.reference_point = reference_point
        self.mean = mean
        self.std = std

    def get_hypervolume(self, frontier: np.ndarray) -> float:
        return calculate_hypervolume(normalize(frontier, self.mean, self.std), self.reference_point)

    def evaluate(self, model: AbstractModel, xs: np.ndarray, ys: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        means, var = model.predict_batch(candidates)
        candidate_values = means - self.gain * np.sqrt(np.maximum(var, self.NOISE))

        frontier_values = get_frontier(ys)
        current_hv = self.get_hypervolume(frontier_values)

        values = np.zeros((candidates.shape[0], 1))
        for i, candidate_value in enumerate(candidate_values):
            penalty = 0.0
            for frontier_value in frontier_values:
                # If frontier value is weakly dominating
                if np.all(frontier_value <= candidate_value + self.epsilon):
                    p = -1 + np.prod(1 + np.maximum(candidate_value - frontier_value, np.zeros_like(candidate_value)))
                    # TODO: Why not sum over all frontier points? penalty += p
                    penalty = np.maximum(penalty, p)

            if penalty == 0.0:
                potential_hv = self.get_hypervolume(np.vstack((candidate_value, frontier_values)))
                value = potential_hv - current_hv
            else:
                value = -penalty

            values[i] = value

        return values
