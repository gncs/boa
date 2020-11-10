import datetime
import sys
from typing import Tuple

import tensorflow as tf
import numpy as np

from boa.acquisition.abstract import AbstractAcquisition
from boa.optimization.data import FileHandler, Data
from boa.models.multi_output_gp_regression_model import MultiOutputGPRegressionModel
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

    def optimize(self,
                 f: AbstractObjective,
                 model: MultiOutputGPRegressionModel,
                 acq_fun: AbstractAcquisition,
                 xs: np.array,
                 ys: np.array,
                 candidate_xs: np.array,
                 optimizer_restarts: int,
                 initialization: str,
                 iters: int,
                 fit_joint: bool,
                 model_optimizer: str,
                 map_estimate: bool,
                 denoising: bool,
                 marginalize_hyperparameters: bool,
                 mcmc_kwargs: dict = {}) -> Tuple[np.ndarray, np.ndarray]:

        xs = xs.copy()
        ys = ys.copy()
        candidate_xs = candidate_xs.copy()

        if self.verbose:
            print_message('Starting optimization')

        for iteration in range(self.max_num_iterations):
            if self.verbose:
                print_message('Iteration: {}'.format(iteration + 1))

            eval_points = []

            eval_model = model.copy()

            # Collect evaluation points
            for _ in range(self.batch_size):

                # Ensure that there are candidates left
                if candidate_xs.size == 0:
                    break

                acquisition_values, y_preds = acq_fun.evaluate(model=eval_model,
                                                               xs=xs,
                                                               ys=ys,
                                                               candidate_xs=candidate_xs)
                max_acquisition_index = np.argmax(acquisition_values)
                eval_point = candidate_xs[max_acquisition_index]
                y_pred = y_preds[max_acquisition_index]

                if self.verbose:
                    print(f"Max acq_value: {np.max(acquisition_values)}")
                    print(f"Min acq_value: {np.min(acquisition_values)}")
                    print(f"median acq: {np.percentile(acquisition_values, 50)}")
                    print(f"25 acq: {np.percentile(acquisition_values, 25)}")
                    print(f"75 acq: {np.percentile(acquisition_values, 75)}")
                    print(f"acq shape {acquisition_values.shape}")
                    print(f"eval shape {candidate_xs.shape}\n")

                    print(f"Eval point: \n{eval_point}\n")
                    print(f"prediction at eval point:\n{y_pred}")
                    print("=================================================")

                eval_points.append(eval_point)

                eval_model = eval_model.condition_on(eval_point, y_pred)

                candidate_xs = np.delete(candidate_xs, max_acquisition_index, axis=0)

            # If no new evaluation points are selected, break
            if not eval_points:
                break

            # Evaluate function at chosen points
            inp, outp = f.evaluate_batch(np.array(eval_points))

            # Add evaluations to data set and model
            xs = np.vstack((xs, inp))
            ys = np.vstack((ys, outp))

            for i, o in zip(inp, outp):
                model = model.condition_on(i.reshape((1, -1)), o.reshape((1, -1)))

            if marginalize_hyperparameters or map_estimate:
                model.initialize_hyperparameters(length_scale_init_mode=initialization)

            if not marginalize_hyperparameters:
                try:
                    print("Fitting model to new data!")
                    model.fit(optimizer_restarts=optimizer_restarts,
                              optimizer=model_optimizer,
                              initialization=initialization,
                              iters=iters,
                              denoising=denoising,
                              map_estimate=map_estimate,
                              fit_joint=fit_joint)
                except Exception as e:
                    print_message('Error: ' + str(e))
                    if not self.strict:
                        print_message('Failed to update model, continuing.')
                    else:
                        raise e

            if self.create_checkpoints:
                self.create_checkpoint(f=f, xs=xs, ys=ys)

        if self.verbose:
            print_message('Finished optimization')

        return xs, ys

    def create_checkpoint(self, f: AbstractObjective, xs: np.ndarray, ys: np.ndarray):
        handler = FileHandler(path=self.LOG_FILE)
        progress = Data(xs=xs, ys=ys, x_labels=f.get_input_labels(), y_labels=f.get_output_labels())
        handler.save(progress)