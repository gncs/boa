from typing import List, Tuple, Optional

import numpy as np

from boa.models.abstract_model import AbstractModel
from .abstract import AbstractAcquisition
from .util import calculate_hypervolume, get_frontier


class SMSEGO(AbstractAcquisition):
    NOISE = 1E-10

    def __init__(
            self,
            gain: float,
            epsilon: float,
            reference: List[float],
            output_slice: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()

        self.gain = gain
        self.epsilon = epsilon
        self.reference = np.array(reference)
        self.output_slice = output_slice

    def slice_output(self, ys: np.ndarray):
        # Slice portion of output to be considered in acquisition function
        if self.output_slice:
            return ys[:, self.output_slice[0]:self.output_slice[1]]
        return ys

    def evaluate(self, model: AbstractModel, xs: np.ndarray, ys: np.ndarray, candidate_xs: np.ndarray) -> np.ndarray:
        ys = self.slice_output(ys)

        # Model predictions for candidates
        means, var = model.predict(candidate_xs, numpy=True)

        means = self.slice_output(means)
        var = self.slice_output(var)

        candidate_ys = means - self.gain * np.sqrt(np.maximum(var, self.NOISE))

        # Normalize so that objectives are treated on equal footing
        ys_mean = np.mean(ys, axis=0)
        ys_std = np.mean(ys, axis=0)

        ys_normalized = (ys - ys_mean) / ys_std
        reference_normalized = (self.reference - ys_mean) / ys_std

        # Normalize
        candidate_ys_normalized = (candidate_ys - ys_mean) / ys_std
        frontier_normalized = get_frontier(ys_normalized)

        current_hv = calculate_hypervolume(points=frontier_normalized, reference=reference_normalized)

        values = np.zeros((candidate_ys_normalized.shape[0], 1))
        for i, candidate in enumerate(candidate_ys_normalized):
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
