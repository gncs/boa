import datetime
import sys
from typing import Tuple

import numpy as np

from boa.acquisition.abstract_v2 import AbstractAcquisition
from boa.optimization.data import FileHandler, Data
from boa.models.abstract_model_v2 import AbstractModel
from boa.objective.abstract import AbstractObjective


def print_message(msg: str):
    date_format = '%Y-%m-%d %H:%M:%S'
    date = datetime.datetime.now().strftime(date_format)
    print('{date}  {msg}'.format(date=date, msg=msg))
    sys.stdout.flush()


class Optimizer:
    LOG_FILE = 'opt_ckpt.json'

    def __init__(self, max_num_iterations: int, batch_size: int, strict=False, checkpoints=True, verbose=False):
        self.max_num_iterations = max_num_iterations
        self.batch_size = batch_size
        self.strict = strict
        self.verbose = verbose
        self.create_checkpoints = checkpoints

    def optimize(self, f: AbstractObjective, model: AbstractModel, acq_fun: AbstractAcquisition, xs: np.array,
                 ys: np.array, candidate_xs: np.array) -> Tuple[np.ndarray, np.ndarray]:

        xs = xs.copy()
        ys = ys.copy()
        candidate_xs = candidate_xs.copy()

        if self.verbose:
            print_message('Starting optimization')

        for iteration in range(self.max_num_iterations):
            if self.verbose:
                print_message('Iteration: {}'.format(iteration + 1))

            eval_points = []

            # Collect evaluation points
            for _ in range(self.batch_size):

                # Ensure that there are candidates left
                if candidate_xs.size == 0:
                    break

                acquisition_values = acq_fun.evaluate(model=model, xs=xs, ys=ys, candidate_xs=candidate_xs)
                max_acquisition_index = np.argmax(acquisition_values)
                eval_point = candidate_xs[max_acquisition_index]

                eval_points.append(eval_point)
                model.add_pseudo_point(eval_point.reshape((1, -1)))

                candidate_xs = np.delete(candidate_xs, max_acquisition_index, axis=0)

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
                model.add_true_point(i.reshape((1, -1)), o.reshape(1, -1))

            try:
                model.train()
            except Exception as e:
                print_message('Error: ' + str(e))
                if not self.strict:
                    print_message('Failed to update model, continuing.')
                else:
                    raise

            if self.create_checkpoints:
                self.create_checkpoint(f=f, xs=xs, ys=ys)

        if self.verbose:
            print_message('Finished optimization')

        return xs, ys

    def create_checkpoint(self, f: AbstractObjective, xs: np.ndarray, ys: np.ndarray):
        handler = FileHandler(path=self.LOG_FILE)
        progress = Data(xs=xs, ys=ys, x_labels=f.get_input_labels(), y_labels=f.get_output_labels())
        handler.save(progress)