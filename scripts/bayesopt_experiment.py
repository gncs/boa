import os

from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from boa.objective.abstract import AbstractObjective

from boa.core import transform_df

from boa.models.random import RandomModel
from boa.models.fully_factorized_gp import FullyFactorizedGPModel
from boa.models.gpar import GPARModel
from boa.models.matrix_factorized_gpar import MatrixFactorizedGPARModel

from boa.acquisition.smsego import SMSEGO

from boa.optimization.data import Data, generate_data, FileHandler
from boa.optimization.optimizer import Optimizer

from boa import ROOT_DIR

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

import datetime

from dataset_config import prepare_ff_gp_data, prepare_gpar_data, load_dataset, dataset_ingredient

ex = Experiment("bayesopt_experiment", ingredients=[dataset_ingredient])
database_url = "127.0.0.1:27017"
database_name = "boa_bayesopt_experiments"
ex.captured_out_filter = apply_backspaces_and_linefeeds

ex.observers.append(MongoObserver(url=database_url,
                                  db_name=database_name))


@ex.config
def bayesopt_config(dataset):
    task = "fft"
    model = "gpar"
    verbose = True

    # Number of experiments to perform
    rounds = 5

    # BayesOpt iterations
    max_num_iterations = 120
    warmup_dataset_size = 25
    batch_size = 1

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Path to the directory to which we will write the log files
    log_dir = f"{ROOT_DIR}/../logs/{task}/bayesopt/"
    save_dir = f"{ROOT_DIR}/../models/{task}/bayesopt/{current_time}/"

    # GP kernel to use.
    kernel = "matern52"

    use_input_transforms = True
    use_output_transforms = False

    map_estimate = False

    marginalize_hyperparameters = False

    if marginalize_hyperparameters:
        num_samples = 50
        num_burnin_steps = 100

        leapfrog_steps = 10
        step_size = 0.03

        mcmc_kwargs = {
            "num_samples": num_samples,
            "num_burnin_steps": num_burnin_steps,
            "leapfrog_steps": leapfrog_steps,
            "step_size": step_size,
    }

    # Number of random initializations to try in a single training cycle.
    model_optimizer_restarts = 3

    # Optimization algorithm to use when fitting the models' hyperparameters.
    model_optimizer = "l-bfgs-b"

    # Initialization heuristic for the hyperparameters of the models.
    initialization = "l2_median"

    # Number of training iterations to allow either for L-BFGS-B or Adam.
    iters = 1000

    matrix_factorized = False
    fit_joint = False

    if model == "ff-gp":
        log_path = f"{log_dir}/{model}/{current_time}/{{}}.json"

    elif model == "gpar":
        log_path = f"{log_dir}/{model}/{current_time}/{{}}.json"

    elif model == "mf-gpar":
        fit_joint = True
        # Effective dimension of the factorization.
        latent_dim = 5
        matrix_factorized = True
        log_path = f"{log_dir}/{model}-{latent_dim}/{current_time}/{{}}.json"

    elif model == "random":
        num_samples = 10

        log_path = f"{log_dir}/{model}-{num_samples}/{current_time}/{{}}.json"


tf.config.experimental.set_visible_devices([], 'GPU')


class Objective(AbstractObjective):
    def __init__(self,
                 df: pd.DataFrame,
                 input_labels: List[str],
                 output_labels: List[str],
                 *args,
                 **kwargs):
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

        if mask.sum() != 1:
            raise Exception(f"sum was {mask.sum()}, value was {value}")
        return self.data.loc[mask, self.output_labels].values


@ex.capture
def optimize(objective,
             model,
             model_optimizer,
             model_optimizer_restarts,
             initialization,
             iters,
             fit_joint,
             optimizer,
             warmup_dataset_size,
             acq,
             seed: int,
             verbose,
             map_estimate,
             marginalize_hyperparameters,
             mcmc_kwargs={}) -> Data:
    np.random.seed(seed)
    tf.random.set_seed(seed)

    candidates = objective.get_candidates()

    data = generate_data(objective=objective, size=warmup_dataset_size, seed=seed)

    model = model.condition_on(xs=data.input, ys=data.output)

    if map_estimate or marginalize_hyperparameters:
        model.initialize_hyperpriors_and_bijectors(length_scale_init_mode=initialization)

    if not marginalize_hyperparameters:
        model.fit(optimizer_restarts=model_optimizer_restarts,
                  optimizer=model_optimizer,
                  iters=iters,
                  fit_joint=fit_joint,
                  length_scale_init_mode=initialization,
                  map_estimate=map_estimate,
                  trace=verbose)

    xs, ys = optimizer.optimize(f=objective,
                                model=model,
                                acq_fun=acq,
                                xs=data.input,
                                ys=data.output,
                                candidate_xs=candidates,
                                fit_joint=fit_joint,
                                iters=iters,
                                initialization=initialization,
                                model_optimizer=model_optimizer,
                                optimizer_restarts=3,
                                map_estimate=map_estimate,
                                marginalize_hyperparameters=marginalize_hyperparameters,
                                mcmc_kwargs=mcmc_kwargs)

    return Data(xs=xs, ys=ys, x_labels=objective.input_labels, y_labels=objective.output_labels)


def get_default_acq_config(df: pd.DataFrame, objective_labels) -> dict:
    max_values = df[objective_labels].apply(max).values

    return {'gain': 1, 'epsilon': 0.01, 'reference': max_values, 'output_slice': (-3, None)}


@ex.automain
def main(dataset,

         rounds,
         model,
         kernel,
         max_num_iterations,
         use_input_transforms,
         batch_size,
         verbose,
         log_path,
         _seed,
         _log,
         latent_dim=None,
         num_samples=None):

    targets = dataset["targets"]
    input_labels = dataset["input_labels"]
    output_labels = dataset["output_labels"]

    ds = load_dataset()

    # Setup acquisition function
    acq_config = get_default_acq_config(ds.df, objective_labels=targets)
    smsego_acq = SMSEGO(**acq_config)

    # Setup optimizer
    bo_optimizer = Optimizer(max_num_iterations=max_num_iterations,
                             batch_size=batch_size,
                             verbose=verbose)

    # Setup models
    if model == "random":
        df = prepare_ff_gp_data(ds)

        surrogate_model = RandomModel(input_dim=len(input_labels),
                                      output_dim=len(output_labels),
                                      seed=_seed,
                                      num_samples=num_samples,
                                      verbose=verbose)

    if model == "ff-gp":
        df = prepare_ff_gp_data(ds)

        surrogate_model = FullyFactorizedGPModel(kernel=kernel,
                                                 input_dim=len(input_labels),
                                                 output_dim=len(output_labels),
                                                 verbose=verbose)
    elif model in ["gpar", "mf-gpar"]:
        df = prepare_gpar_data(ds, targets=targets)
        if model == "gpar":
            surrogate_model = GPARModel(kernel=kernel,
                                        input_dim=len(input_labels),
                                        output_dim=len(output_labels),
                                        verbose=verbose)

        elif model == "mf-gpar":
            surrogate_model = MatrixFactorizedGPARModel(kernel=kernel,
                                                        input_dim=len(input_labels),
                                                        output_dim=len(output_labels),
                                                        latent_dim=latent_dim,
                                                        verbose=verbose)
    if use_input_transforms:
        df = transform_df(df, dataset["input_transforms"])

    # Run the optimization
    for seed in range(rounds):
        results = optimize(objective=Objective(df=df,
                                               input_labels=input_labels,
                                               output_labels=output_labels),
                           model=surrogate_model,
                           optimizer=bo_optimizer,
                           acq=smsego_acq,
                           seed=seed,
                           verbose=verbose)

        os.makedirs(os.path.dirname(log_path.format(seed)), exist_ok=True)
        _log.info(f"Saving model to {log_path.format(seed)}!")
        handler = FileHandler(path=log_path.format(seed))
        handler.save(results)
