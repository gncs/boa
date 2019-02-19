from typing import Tuple

import numpy as np

from boa.acquisition.abstract import AbstractAcquisition
from boa.models.abstract import AbstractModel
from boa.objective.abstract import AbstractObjective
from boa.util import print_message


class Optimizer:
    def __init__(self, max_num_iterations: int, batch_size: int, verbose: bool = False):
        self.max_num_iterations = max_num_iterations
        self.batch_size = batch_size
        self.verbose = verbose

    def optimize(self, f: AbstractObjective, model: AbstractModel, acq_fun: AbstractAcquisition, xs: np.array,
                 ys: np.array, candidates: np.array) -> Tuple[np.ndarray, np.ndarray]:

        xs = xs.copy()
        ys = ys.copy()

        if self.verbose:
            print_message('Starting optimization')

        for iteration in range(self.max_num_iterations):
            if self.verbose:
                print_message('Iteration: {}'.format(iteration + 1))

            eval_points = []

            # Collect evaluation points
            for _ in range(self.batch_size):
                # Ensure that there are candidates left
                if candidates.size == 0:
                    break

                acquisition_values = acq_fun.evaluate(model=model, xs=xs, ys=ys, candidates=candidates)
                max_acquisition_index = np.argmax(acquisition_values)
                eval_point = candidates[max_acquisition_index]

                eval_points.append(eval_point)
                model.add_pseudo_point(eval_point)

                candidates = np.delete(candidates, max_acquisition_index, axis=0)

            model.remove_pseudo_points()

            # If no new evaluation points are selected, break
            if not eval_points:
                break

            # Evaluate function at chosen points
            inp, outp = f.evaluate_batch(np.array(eval_points))

            # Add evaluations to data set and model
            xs = np.vstack((xs, inp))
            ys = np.vstack((ys, outp))

            for i, o in zip(inp, outp):
                model.add_true_point(i, o)

            model.train()

        if self.verbose:
            print_message('Finished optimization')

        return xs, ys
