import json
import time
import os

import numpy as np

from sklearn.model_selection import train_test_split

from boa.models.fully_factorized_gp import FullyFactorizedGPModel
from boa.models.gpar import GPARModel
from boa.models.random import RandomModel
from boa.models.matrix_factorized_gpar import MatrixFactorizedGPARModel

from boa.core import transform_df, back_transform
from boa import ROOT_DIR

import tensorflow as tf

import datetime
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from dataset_config import dataset_ingredient, load_dataset

ex = Experiment("fitting_experiment", ingredients=[dataset_ingredient])
database_url = "127.0.0.1:27017"
database_name = "boa_fitting_experiments"
ex.captured_out_filter = apply_backspaces_and_linefeeds

ex.observers.append(MongoObserver(url=database_url,
                                  db_name=database_name))


@ex.config
def experiment_config(dataset):
    task = dataset["name"]
    model = "gpar"
    verbose = True

    use_input_transforms = True
    use_output_transforms = False

    map_estimate = False

    # Number of experiments to perform
    rounds = 5

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Path to the directory to which we will write the log files
    log_dir = f"{ROOT_DIR}/../logs/{task}/pred/"
    save_dir = f"{ROOT_DIR}/../models/{task}/pred/{current_time}/"

    # GP kernel to use.
    kernel = "matern52"

    marginalize_hyperparameters = False
    empirical_bayes_for_marginalization = False

    mcmc_kwargs = {}

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
    num_optimizer_restarts = 5

    # Optimization algorithm to use when fitting the models' hyperparameters.
    optimizer = "l-bfgs-b"

    # Initialization heuristic for the hyperparameters of the models.
    initialization = "l2_median"

    # Number of training iterations to allow either for L-BFGS-B or Adam.
    iters = 1000

    matrix_factorized = False

    fit_joint = False

    if model in ["ff-gp", "gpar", "random"]:
        log_dir = f"{log_dir}/{model}/{current_time}/"
        log_path = f"{log_dir}/{model}_experiments.json"

    elif model == "mf-gpar":
        fit_joint = True
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
                   map_estimate,
                   initialization,
                   num_optimizer_restarts,
                   use_input_transforms,
                   use_output_transforms,
                   log_dir,
                   log_path,
                   save_dir,
                   fit_joint,
                   matrix_factorized,
                   iters: int,
                   rounds,
                   verbose,
                   marginalize_hyperparameters,
                   empirical_bayes_for_marginalization,
                   _log,
                   _seed,
                   mcmc_kwargs={}):

    input_labels = dataset["input_labels"]
    output_labels = dataset["output_labels"]

    input_transforms = dataset["input_transforms"]
    output_transforms = dataset["output_transforms"]

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
                          'inputs': input_labels,
                          'outputs': output_labels,
                          'input_transformed': use_input_transforms,
                          'output_transformed': use_output_transforms}

            if matrix_factorized:
                experiment["latent_size"] = model.latent_dim

            train, test = train_test_split(data,
                                           train_size=size,
                                           test_size=200,
                                           random_state=_seed + index)

            ys_transforms = None

            # Transform inputs
            if use_input_transforms:
                train = transform_df(train, input_transforms)
                test = transform_df(test, input_transforms)

            # Transform outputs
            if use_output_transforms:
                train = transform_df(train, output_transforms)

                ys_transforms = [(output_transforms[k]
                                  if k in output_transforms
                                  else None)
                                 for k in output_labels]
                print(ys_transforms)

            start_time = time.time()
            try:
                model = model.condition_on(train[input_labels].values,
                                           train[output_labels].values,
                                           keep_previous=False)

                if map_estimate:
                    model.initialize_hyperpriors(length_scale_init_mode=initialization)

                if not marginalize_hyperparameters or empirical_bayes_for_marginalization:
                    model.fit(ys_transforms=ys_transforms,
                              fit_joint=fit_joint,
                              map_estimate=map_estimate,
                              length_scale_init_mode=initialization,
                              optimizer_restarts=num_optimizer_restarts,
                              optimizer=optimizer,
                              trace=True,
                              err_level="raise",
                              iters=iters)
            except Exception as e:
                _log.exception("Training failed: {}".format(str(e)))
                raise e

            experiment['train_time'] = time.time() - start_time

            save_path = f"{save_dir}/size_{size}/model_{index}/model"
            model.save(save_path, )
            _log.info(f"Saved model to {save_path}!")

            start_time = time.time()

            try:
                if marginalize_hyperparameters:
                    model.initialize_hyperpriors(length_scale_init_mode=initialization,
                                                 empirical_bayes=empirical_bayes_for_marginalization)

                mean, variance = model.predict(test[input_labels].values,
                                               marginalize_hyperparameters=marginalize_hyperparameters,
                                               numpy=True,
                                               **mcmc_kwargs)

                # Back-transfrom predictions!
                # *Note*: If we are using a log-transform, the back-transformed mean is actually
                # Going to be the median, exp(mu) NOT the expected value exp(mu + var/2)!
                if use_output_transforms:
                    mean, variance = back_transform(mean, variance,
                                                    output_labels,
                                                    output_transforms)

            except Exception as e:
                _log.exception("Prediction failed: {}, saving model!".format(str(e)))
                raise e

            experiment['predict_time'] = time.time() - start_time

            diff = (test[output_labels].values - mean)

            experiment['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
            experiment['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
            experiment['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()

            experiment['std_mean_squ_err'] = np.mean(np.square(diff) / variance, axis=0).tolist()
            experiment['std_mean_abs_err'] = np.mean(np.abs(diff) / np.sqrt(variance + 1e-7), axis=0).tolist()
            experiment['mean_predictive_variance'] = np.mean(variance)
            experiment['mean_predictive_std'] = np.mean(np.sqrt(variance + 1e-7))

            experiments.append(experiment)

            _log.info("Saving experiments to {}".format(log_path))
            with open(log_path, mode='w') as out_file:
                json.dump(experiments, out_file, sort_keys=True, indent=4)

    return experiments


@ex.automain
def main(dataset, model, kernel, verbose, latent_dim=None):
    data = load_dataset()

    input_dim = len(dataset["input_labels"])
    output_dim = len(dataset["output_labels"])

    if model == "random":
        surrogate_model = RandomModel(input_dim=input_dim,
                                      output_dim=output_dim,
                                      seed=42,
                                      num_samples=50)

    elif model == 'ff-gp':
        surrogate_model = FullyFactorizedGPModel(kernel=kernel,
                                                 input_dim=input_dim,
                                                 output_dim=output_dim,
                                                 verbose=verbose)

    elif model == 'gpar':
        surrogate_model = GPARModel(kernel=kernel,
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    verbose=verbose)

    elif model == 'mf-gpar':
        surrogate_model = MatrixFactorizedGPARModel(kernel=kernel,
                                                    input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    latent_dim=latent_dim,
                                                    verbose=verbose)

    else:
        raise NotImplementedError

    run_experiment(model=surrogate_model,
                   data=data.df)
