import logging
import argparse
import json
import time
import os
from typing import Sequence

import numpy as np

from sklearn.model_selection import train_test_split

from boa.models.fully_factorized_gp import FullyFactorizedGPModel
from boa.models.gpar import GPARModel
from boa.models.matrix_factorized_gpar import MatrixFactorizedGPARModel
from boa.models.gpar_perm import PermutedGPARModel

from boa.core import GaussianProcess, transform_df, back_transform
from boa import ROOT_DIR

import tensorflow as tf

import datetime
from sacred import Experiment

from dataset_config import dataset_ingredient, load_dataset
from dataset_config import prepare_gpar_data, prepare_ff_gp_data, prepare_ff_gp_aux_data

ex = Experiment("fitting_experiment", ingredients=[dataset_ingredient])


@ex.config
def experiment_config(dataset):
    task = "fft"
    model = "gpar"
    verbose = True

    use_input_transforms = True
    use_output_transforms = False

    # Number of experiments to perform
    rounds = 5

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Path to the directory to which we will write the log files
    log_dir = f"{ROOT_DIR}/../logs/{task}/pred/"
    save_dir = f"{ROOT_DIR}/../models/{task}/pred/{current_time}/"

    # GP kernel to use.
    kernel = "matern52"

    # Number of random initializations to try in a single training cycle.
    num_optimizer_restarts = 5

    # Optimization algorithm to use when fitting the models' hyperparameters.
    optimizer = "l-bfgs-b"

    # Initialization heuristic for the hyperparameters of the models.
    initialization = "dim_median"

    # Number of training iterations to allow either for L-BFGS-B or Adam.
    iters = 1000

    matrix_factorized = False

    if model == "ff-gp":
        log_dir = f"{log_dir}/{model}/{current_time}/"
        log_path = f"{log_dir}/{model}_experiments.json"

    elif model == "gpar":
        log_dir = f"{log_dir}/{model}/{current_time}/"
        log_path = f"{log_dir}/{model}_experiments.json"

    elif model == "mf_gpar":
        # Effective dimension of the factorization.
        latent_dim = 5
        matrix_factorized = True

        log_dir = f"{log_dir}/{model}-{latent_dim}/{current_time}/"
        log_path = f"{log_dir}/{model}-{latent_dim}_experiments.json"


AVAILABLE_DATASETS = ["fft", "stencil3d", "gemm", "smaug"]
AVAILABLE_OPTIMIZERS = ["l-bfgs-b", "adam"]
AVAILABLE_INITIALIZATION = ["median", "random", "dim_median"]

# Set CPU as available physical device
tf.config.experimental.set_visible_devices([], 'GPU')


@ex.capture
def run_experiment(model,
                   data,

                   dataset,
                   optimizer,
                   num_optimizer_restarts,
                   use_input_transforms,
                   use_output_transforms,
                   log_dir,
                   log_path,
                   save_dir,
                   matrix_factorized,
                   iters: int,
                   rounds,
                   verbose,
                   _log,
                   _seed):
    # Make sure the directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    experiments = []

    # Set seed for reproducibility
    np.random.seed(_seed)
    tf.random.set_seed(_seed)

    for size in [25, 50, 100, 150, 200]:
        for index in range(rounds):

            if verbose:
                _log.info("-----------------------------------------------------------")
                _log.info(f"Training round: {index + 1} for training set size {size}")
                _log.info("-----------------------------------------------------------")

            experiment = {'index': index,
                          'size': size,
                          'inputs': dataset["input_labels"],
                          'outputs': dataset["output_labels"],
                          'input_transformed': use_input_transforms,
                          'output_transformed': use_output_transforms}

            if matrix_factorized:
                experiment["latent_size"] = model.latent_dim

            train, test = train_test_split(data, train_size=size, test_size=200, random_state=_seed + index)

            # Transform inputs
            if use_input_transforms:
                train = transform_df(train, dataset["input_transforms"])
                test = transform_df(test, dataset["input_transforms"])

            # Transform outputs
            if use_output_transforms:
                train = transform_df(train, dataset["output_transforms"])

            start_time = time.time()
            try:
                model = model.condition_on(train[dataset["input_labels"]].values,
                                           train[dataset["output_labels"]].values[:, :],
                                           keep_previous=False)
                model.fit_to_conditioning_data(optimizer_restarts=num_optimizer_restarts,
                                               optimizer=optimizer,
                                               trace=True,
                                               err_level="raise",
                                               iters=iters)
            except Exception as e:
                _log.exception("Training failed: {}".format(str(e)))
                raise e

            experiment['train_time'] = time.time() - start_time

            save_path = f"{save_dir}/size_{size}/model_{index}/model"
            model.save(save_path)
            _log.info(f"Saved model to {save_path}!")

            start_time = time.time()

            try:
                mean, variance = model.predict(test[dataset["input_labels"]].values, numpy=True)

                # Back-transfrom predictions!
                # *Note*: If we are using a log-transform, the back-transformed mean is actually
                # Going to be the median, exp(mu) NOT the expected value exp(mu + var/2)!
                if use_output_transforms:
                    mean, variance = back_transform(mean, variance,
                                                    dataset["output_labels"],
                                                    dataset["output_transforms"])

            except Exception as e:
                _log.exception("Prediction failed: {}, saving model!".format(str(e)))
                raise e

            experiment['predict_time'] = time.time() - start_time

            diff = (test[dataset["output_labels"]].values - mean)

            experiment['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
            experiment['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
            experiment['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()

            experiment['std_mean_squ_err'] = np.mean(np.square(diff) / variance, axis=0).tolist()
            experiment['std_mean_abs_err'] = np.mean(np.abs(diff) / np.sqrt(variance + 1e-7), axis=0).tolist()

            experiments.append(experiment)

            _log.info("Saving experiments to {}".format(log_path))
            with open(log_path, mode='w') as out_file:
                json.dump(experiments, out_file, sort_keys=True, indent=4)

    return experiments


@ex.automain
def main(dataset, model, kernel, initialization, verbose, latent_dim=None):
    data = load_dataset()

    if model == 'ff-gp':
        df = prepare_ff_gp_data(data)

        surrogate_model = FullyFactorizedGPModel(kernel=kernel,
                                                 input_dim=len(dataset["input_labels"]),
                                                 output_dim=len(dataset["output_labels"]),
                                                 initialization_heuristic=initialization,
                                                 verbose=verbose)

    elif model in ["gpar", "mf-gpar", "p-gpar"]:
        df = prepare_gpar_data(data)

        if model == 'gpar':
            surrogate_model = GPARModel(kernel=kernel,
                                        input_dim=len(dataset["input_labels"]),
                                        output_dim=len(dataset["output_labels"]),
                                        initialization_heuristic=initialization,
                                        verbose=verbose)

        elif model == 'mf-gpar':
            surrogate_model = MatrixFactorizedGPARModel(kernel=kernel,
                                                        input_dim=len(dataset["input_labels"]),
                                                        output_dim=len(dataset["output_labels"]),
                                                        latent_dim=latent_dim,
                                                        initialization_heuristic=initialization,
                                                        verbose=verbose)

        elif model == 'p-gpar':
            surrogate_model = PermutedGPARModel(kernel=kernel,
                                                input_dim=len(dataset["input_labels"]),
                                                output_dim=len(dataset["output_labels"]),
                                                initialization_heuristic=initialization,
                                                verbose=verbose)

    results = run_experiment(model=surrogate_model,
                             data=df)
