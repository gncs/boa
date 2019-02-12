from typing import List

import numpy as np

from boa.models.abstract import AbstractModel
from .abstract import AbstractAcquisition
from .util import calculate_hypervolume, get_frontier


class SMSEGO(AbstractAcquisition):
    NOISE = 1E-10

    def __init__(self, gain: float, epsilon: float, reference_point: List[float]) -> None:
        super().__init__()

        self.gain = gain
        self.epsilon = epsilon
        self.reference_point = np.array(reference_point)

    def evaluate(self, model: AbstractModel, xs: np.ndarray, ys: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        # Model predictions for candidates
        means, var = model.predict_batch(candidates)
        candidates = means - self.gain * np.sqrt(np.maximum(var, self.NOISE))

        # Normalize so that objectives are treated on equal footing
        ys_mean = np.mean(ys, axis=0)
        ys_std = np.mean(ys, axis=0)

        ys_normalized = (ys - ys_mean) / ys_std
        reference_normalized = (self.reference_point - ys_mean) / ys_std

        # Normalize
        candidates_normalized = (candidates - ys_mean) / ys_std
        frontier_normalized = get_frontier(ys_normalized)

        current_hv = calculate_hypervolume(points=frontier_normalized, reference=reference_normalized)

        values = np.zeros((candidates_normalized.shape[0], 1))
        for i, candidate in enumerate(candidates_normalized):
            max_penalty = 0.0
            # Iterate over frontier values and choose maximum value
            for frontier in frontier_normalized:
                # If frontier value is weakly dominating
                if np.all(frontier <= candidate + self.epsilon):
                    penalty = -1 + np.prod(1 + (candidate - frontier))
                    max_penalty = np.maximum(max_penalty, penalty)

            if max_penalty == 0.0:
                potential_hv = calculate_hypervolume(
                    points=np.vstack((candidate, frontier_normalized)),
                    reference=reference_normalized,
                )
                value = potential_hv - current_hv
            else:
                value = -max_penalty

            values[i] = value

        return values
