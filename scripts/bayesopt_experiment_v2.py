import logging
import argparse
import os

from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from boa.core.utils import setup_logger
from boa.core.gp import GaussianProcess

from boa.datasets.loader import load_dataset

from boa.objective.abstract import AbstractObjective

from boa.models.fully_factorized_gp_v2 import FullyFactorizedGPModel
from boa.models.gpar_v2 import GPARModel

from boa.acquisition.smsego_v2 import SMSEGO

from boa.optimization.data import Data, generate_data, FileHandler
from boa.optimization.optimizer_v2 import Optimizer

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/bayesopt_experiment_v2")

OBJECTIVE_TARGETS = {
    "fft": ['cycle', 'avg_power', 'total_area'],
    "stencil3d": []
}

AVAILABLE_DATASETS = ["fft",
                      "stencil3d"]

AVAILABLE_OPTIMIZERS = ["l-bfgs-b", "adam"]

AVAILABLE_INITIALIZATION = ["median", "random", "dim_median"]

DEFAULT_OPTIMIZER_CONFIG = {
    'max_num_iterations': 120,
    'batch_size': 1,
    'verbose': True,
}


class Objective(AbstractObjective):
    def __init__(self, df: pd.DataFrame, input_labels: List[str], output_labels: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = df
        self.input_labels = input_labels
        self.output_labels = output_labels

    def get_input_labels(self) -> List[str]:
        """Return input labels as a list of length D_input"""
        return self.input_labels

    def get_output_labels(self) -> List[str]:
        """Return output labels as a list of length D_output"""
        return self.output_labels

    def get_candidates(self) -> np.ndarray:
        """Return potential candidates as an array of shape N x D_input"""
        return self.data[self.input_labels].values

    def __call__(self, value: np.ndarray) -> np.ndarray:
        """Return output of objective function as an array of shape N x D_output"""
        mask = pd.Series([True] * self.data.shape[0])
        for k, v in zip(self.input_labels, value):
            mask = mask & (self.data[k].values == v)

        assert (mask.sum() == 1)

        return self.data.loc[mask, self.output_labels].values


def optimize(objective,
             model,
             model_optimizer_restarts,
             optimizer,
             acq,
             seed: int) -> Data:

    np.random.seed(seed)
    tf.random.set_seed(seed)

    init_data_size = 10

    candidates = objective.get_candidates()

    data = generate_data(objective=objective,
                         size=init_data_size,
                         seed=seed)

    model = model.condition_on(xs=data.input,
                               ys=data.output)

    model.fit_to_conditioning_data(optimizer_restarts=model_optimizer_restarts)

    xs, ys = optimizer.optimize(
        f=objective,
        model=model,
        acq_fun=acq,
        xs=data.input,
        ys=data.output,
        candidate_xs=candidates
    )

    return Data(xs=xs,
                ys=ys,
                x_labels=objective.input_labels,
                y_labels=objective.output_labels)


def get_default_acq_config(df: pd.DataFrame) -> dict:
    max_values = df[objective_labels].apply(max).values

    return {'gain': 1,
            'epsilon': 0.01,
            'reference': max_values,
            'output_slice': (-3, None)
            }


def prepare_ff_gp_data(data):
    return data.df, data.input_labels.copy(), data.output_labels.copy()


def prepare_gpar_data(data,
                      targets):
    output_labels = data.output_labels.copy()

    for target in targets:
        output_labels.remove(target)
        output_labels.append(target)

    return data.df, data.input_labels.copy(), output_labels


def main(args):
    dataset = load_dataset(path=args.dataset, kind=args.type)

    # Setup acquisition function
    acq_config = get_default_acq_config(dataset)
    smsego_acq = SMSEGO(**acq_config)

    # Setup optimizer
    optimizer = Optimizer(**DEFAULT_OPTIMIZER_CONFIG)

    # Setup models
    if args.model == "ff-gp":
        df, input_labels, output_labels = prepare_ff_gp_data(dataset)

        model = FullyFactorizedGPModel(kernel=args.kernel,
                                       input_dim=len(input_labels),
                                       output_dim=len(output_labels),
                                       initialization_heuristic=args.initialization,
                                       verbose=args.verbose)
    elif args.model == "gpar":
        df, input_labels, output_labels = prepare_gpar_data(dataset,
                                                            targets=OBJECTIVE_TARGETS[args.task])
        model = GPARModel(kernel=args.kernel,
                          input_dim=len(input_labels),
                          output_dim=len(output_labels),
                          initialization_heuristic=args.initialization,
                          verbose=args.verbose)

    # Run the optimization
    for seed in range(5):
        results = optimize(objective=Objective(df=df,
                                               input_labels=input_labels,
                                               output_labels=output_labels),
                           model=model,
                           optimizer=optimizer,
                           acq=smsego_acq,
                           seed=seed)

        save_dir = args.logdir + "/bayesopt/" + args.model
        os.makedirs(save_dir, exist_ok=True)
        handler = FileHandler(path=save_dir + f"/{seed}.json")
        handler.save(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-D', type=str, required=True,
                        help="Path to the dataset.")

    parser.add_argument('--task', '-T', choices=AVAILABLE_DATASETS, required=True,
                        help="Task for which we are providing the dataset.")

    parser.add_argument('--logdir', type=str, default="logs",
                        help="Path to the directory to which we will write the log files "
                             "for the experiment.")

    parser.add_argument('--verbose', action="store_true", default=False,
                        help="Turns on verbose logging")

    model_subparsers = parser.add_subparsers(title="model",
                                             dest="model",
                                             help="Model to fit to the data.")

    model_subparsers.required = True

    # =========================================================================
    # Fully factorized GP
    # =========================================================================

    ff_gp_mode = model_subparsers.add_parser("ff-gp",
                                             formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                             description="Use a fully factorized GP model.")

    # =========================================================================
    # GPAR
    # =========================================================================

    gpar_mode = model_subparsers.add_parser("gpar",
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                            description="Use a GPAR model.")

    # Add common options to models
    for mode in [ff_gp_mode, gpar_mode]:
        mode.add_argument("--kernel",
                          choices=GaussianProcess.AVAILABLE_KERNELS,
                          default="matern52",
                          help="GP kernel to use.")

        mode.add_argument("--num_optimizer_restarts",
                          type=int,
                          default=5,
                          help="Number of random initializations to try in a single training cycle.")

        mode.add_argument("--optimizer",
                          choices=AVAILABLE_OPTIMIZERS,
                          default=AVAILABLE_OPTIMIZERS[0],
                          help="Optimization algorithm to use when fitting the models' hyperparameters.")

        mode.add_argument("--initialization",
                          choices=AVAILABLE_INITIALIZATION,
                          default=AVAILABLE_INITIALIZATION[0],
                          help="Initialization heuristic for the hyperparameters of the models.")
