import logging
import argparse
import os

from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from boa.core.gp import GaussianProcess

from boa.objective.abstract import AbstractObjective

from boa.models.random import RandomModel
from boa.models.fully_factorized_gp import FullyFactorizedGPModel
from boa.models.gpar import GPARModel
from boa.models.matrix_factorized_gpar import MatrixFactorizedGPARModel

from boa.acquisition.smsego import SMSEGO

from boa.optimization.data import Data, generate_data, FileHandler
from boa.optimization.optimizer import Optimizer

from boa import ROOT_DIR

from sacred import Experiment
import datetime

from dataset_config import prepare_ff_gp_data, prepare_gpar_data, load_dataset, dataset_ingredient

ex = Experiment("bayesopt_experiment", ingredients=[dataset_ingredient])


@ex.config
def bayesopt_config(dataset):
    task = "fft"
    model = "gpar"
    verbose = True

    # Number of experiments to perform
    rounds = 5

    # BayesOpt iterations
    max_num_iterations = 120
    batch_size = 1

    targets = dataset.targets

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Path to the directory to which we will write the log files
    log_dir = f"{ROOT_DIR}/../logs/{task}/bayesopt/"
    save_dir = f"{ROOT_DIR}/../models/{task}/bayesopt/{current_time}/"

    # GP kernel to use.
    kernel = "matern52"

    # Number of random initializations to try in a single training cycle.
    num_optimizer_restarts = 5

    # Optimization algorithm to use when fitting the models' hyperparameters.
    optimizer = "l-bfgs-b"

    # Initialization heuristic for the hyperparameters of the models.
    initialization = "median"

    # Number of training iterations to allow either for L-BFGS-B or Adam.
    iters = 1000

    matrix_factorized = False

    if model == "ff-gp":
        log_path = f"{log_dir}/{model}/{current_time}/{{}}.json"

    elif model == "gpar":
        log_path = f"{log_dir}/{model}/{current_time}/{{}}.json"

    elif model == "mf_gpar":
        # Effective dimension of the factorization.
        latent_dim = 5
        matrix_factorized = True
        log_path = f"{log_dir}/{model}-{latent_dim}/{current_time}/{{}}.json"

    elif model == "random":
        num_samples = 10

        log_path = f"{log_dir}/{model}-{num_samples}/{current_time}/{{}}.json"


AVAILABLE_OPTIMIZERS = ["l-bfgs-b", "adam"]

AVAILABLE_INITIALIZATION = ["median", "random", "dim_median"]

tf.config.experimental.set_visible_devices([], 'GPU')


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


def optimize(objective, model, model_optimizer, model_optimizer_restarts, optimizer, acq, seed: int, verbose) -> Data:
    np.random.seed(seed)
    tf.random.set_seed(seed)

    init_data_size = 10

    candidates = objective.get_candidates()

    data = generate_data(objective=objective, size=init_data_size, seed=seed)

    model = model.condition_on(xs=data.input, ys=data.output)

    model.fit_to_conditioning_data(optimizer_restarts=model_optimizer_restarts,
                                   optimizer=model_optimizer,
                                   trace=verbose)

    xs, ys = optimizer.optimize(f=objective,
                                model=model,
                                acq_fun=acq,
                                xs=data.input,
                                ys=data.output,
                                candidate_xs=candidates,
                                optimizer_restarts=3)

    return Data(xs=xs, ys=ys, x_labels=objective.input_labels, y_labels=objective.output_labels)


def get_default_acq_config(df: pd.DataFrame, objective_labels) -> dict:
    max_values = df[objective_labels].apply(max).values

    return {'gain': 1, 'epsilon': 0.01, 'reference': max_values, 'output_slice': (-3, None)}


@ex.automain
def main(model,
         kernel,
         initialization,
         targets,
         optimizer,
         max_num_iterations,
         num_optimizer_restarts,
         batch_size,
         verbose,
         save_dir,
         _seed,
         latent_dim=None,
         num_samples=None):
    dataset = load_dataset()

    # Setup acquisition function
    acq_config = get_default_acq_config(dataset.df, objective_labels=targets)
    smsego_acq = SMSEGO(**acq_config)

    # Setup optimizer
    bo_optimizer = Optimizer(max_num_iterations=max_num_iterations,
                             batch_size=batch_size,
                             verbose=verbose)

    # Setup models
    if model == "random":
        df, input_labels, output_labels = prepare_ff_gp_data(dataset)

        surrogate_model = RandomModel(input_dim=len(input_labels),
                                      output_dim=len(output_labels),
                                      seed=_seed,
                                      num_samples=num_samples,
                                      verbose=verbose)

    if model == "ff-gp":
        df, input_labels, output_labels = prepare_ff_gp_data(dataset)

        surrogate_model = FullyFactorizedGPModel(kernel=kernel,
                                                 input_dim=len(input_labels),
                                                 output_dim=len(output_labels),
                                                 initialization_heuristic=initialization,
                                                 verbose=verbose)
    elif model in ["gpar", "mf-gpar"]:
        df, input_labels, output_labels = prepare_gpar_data(dataset, targets=targets)
        if model == "gpar":
            surrogate_model = GPARModel(kernel=kernel,
                                        input_dim=len(input_labels),
                                        output_dim=len(output_labels),
                                        initialization_heuristic=initialization,
                                        verbose=verbose)

        elif model == "mf-gpar":
            surrogate_model = MatrixFactorizedGPARModel(kernel=kernel,
                                                        input_dim=len(input_labels),
                                                        output_dim=len(output_labels),
                                                        latent_dim=latent_dim,
                                                        initialization_heuristic=initialization,
                                                        verbose=verbose)

    # Run the optimization
    for seed in range(5):
        results = optimize(objective=Objective(df=df, input_labels=input_labels, output_labels=output_labels),
                           model=surrogate_model,
                           model_optimizer=optimizer,
                           model_optimizer_restarts=num_optimizer_restarts,
                           optimizer=bo_optimizer,
                           acq=smsego_acq,
                           seed=seed,
                           verbose=verbose)

        os.makedirs(save_dir, exist_ok=True)
        handler = FileHandler(path=save_dir + f"/{seed}.json")
        handler.save(results)
