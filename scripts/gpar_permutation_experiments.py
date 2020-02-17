import logging
import argparse
import json
import time
import os
from typing import Sequence

import numpy as np

from sklearn.model_selection import train_test_split

from boa.models.gpar import GPARModel
from boa.models.matrix_factorized_gpar import MatrixFactorizedGPARModel
from boa.models.gpar_perm import PermutedGPARModel

from boa.core import GaussianProcess, setup_logger
from boa.core.distribution import GumbelMatching

from boa.datasets.loader import load_dataset

import tensorflow as tf

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/ordering_experiments.log")

AVAILABLE_DATASETS = ["fft", "stencil3d", "gemm"]
AVAILABLE_OPTIMIZERS = ["l-bfgs-b", "adam"]
AVAILABLE_INITIALIZATION = ["median", "random", "dim_median"]

AVAILABLE_RANDOM_SEARCH_OPTIONS = ["linear", "log-linear", "quadratic"]

DATASET_TARGETS = {
    "fft": ('avg_power', 'cycle', 'total_area'),
    "stencil3d": ('avg_power', 'cycle', 'total_area'),
    "gemm": ('avg_power', 'cycle', 'total_area')
}

LOG_LEVELS = {"info": logging.INFO, "debug": logging.DEBUG, "warn": logging.WARNING}

DEFAULT_MODEL_SAVE_DIR = "models/ordiering_experiments/"

# Set CPU as available physical device
tf.config.experimental.set_visible_devices([], 'GPU')


def run_random_experiment(model,
                          data,
                          optimizer,
                          optimizer_restarts,
                          inputs: Sequence[str],
                          outputs: Sequence[str],
                          num_target_dims,
                          training_set_size,
                          validation_set_size,
                          test_set_size,
                          logdir,
                          experiment_file_name,
                          matrix_factorized,
                          num_samples,
                          rounds,
                          seed: int = 42,
                          verbose=False):

    experiment_file_path = os.path.join(logdir, experiment_file_name)

    # Make sure the directory exists
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    experiments = []

    # Set seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Split the data into training and validation set and the held-out test set
    train_and_validate, test = train_test_split(data,
                                                train_size=training_set_size + validation_set_size,
                                                test_size=test_set_size,
                                                random_state=seed)

    # =========================================================================
    # Perform random search
    # =========================================================================
    num_permuted_dimensions = len(outputs) - num_target_dims

    # Create uniform distribution over permutations
    uniform_gm = GumbelMatching(weight_matrix=tf.zeros((num_permuted_dimensions,
                                                        num_permuted_dimensions),
                                                       dtype=tf.float64))

    if num_samples == "linear":
        num_samples = num_permuted_dimensions

    elif num_samples == "log-linear":
        num_samples = tf.math.ceil(tf.math.log(num_permuted_dimensions)) * num_permuted_dimensions

    elif num_samples == "quadratic":
        num_samples = num_permuted_dimensions * num_permuted_dimensions

    else:
        raise Exception(f"Unknown magnitude for the number of random samples to be drawn: {num_samples}!")

    logger.info(f"Performing random order search using {num_samples} samples!")

    # We "restart" the search rounds number of times
    for index in range(rounds):

        experiment = {'index': index,
                      'size': training_set_size,
                      'inputs': inputs,
                      'outputs': outputs}

        # In each round we do a different train-validate split
        train, validate = train_test_split(data,
                                           train_size=training_set_size,
                                           test_size=validation_set_size,
                                           random_state=seed + index)

        # We will record the statistics for every permutation selected, and choose the best one of these
        sample_stats = {}

        for sample_number in range(num_samples):

            # Set the seed for reproducibility in a way that we never use the same seed
            tf.random.set_seed(seed * (index + 1) + sample_number)

            # Draw a new permutation
            perm = uniform_gm.sample(as_tuple=True)

            # convert numpy.int64 to python int
            perm = tuple([p.item() for p in perm])

            # Complete the permutation to all inputs
            perm = perm + tuple(range(num_permuted_dimensions, len(outputs)))

            sample_stat = {"perm": perm}

            logger.info(f"Training round: {index + 1}/{rounds} for training set size {training_set_size}, "
                        f"permutation #{sample_number + 1} out of {num_samples}: {perm}")

            if matrix_factorized:
                sample_stat["latent_size"] = model.latent_dim

            # -----------------------------------------------------------------------------
            # Train model
            # -----------------------------------------------------------------------------
            start_time = time.time()
            try:
                model = model.condition_on(train[inputs].values, train[outputs].values[:, perm], keep_previous=False)
                model.fit_to_conditioning_data(optimizer_restarts=optimizer_restarts, optimizer=optimizer, trace=True)
            except Exception as e:
                logger.exception("Training failed: {}".format(str(e)))
                raise e

            sample_stat['train_time'] = time.time() - start_time

            # -----------------------------------------------------------------------------
            # Validate model
            # -----------------------------------------------------------------------------
            start_time = time.time()
            try:
                mean, _ = model.predict(validate[inputs].values, numpy=True)

                validation_log_prob = model.log_prob(validate[inputs].values,
                                                     validate[outputs].values[:, perm],
                                                     use_conditioning_data=True,
                                                     numpy=True)

            except Exception as e:
                logger.exception("Prediction failed: {}, saving model!".format(str(e)))

                model.save("models/exceptions/" + experiment_file_name + "/model")
                raise e

            sample_stat['predict_time'] = time.time() - start_time

            diff = (validate[outputs].values - mean)

            sample_stat['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
            sample_stat['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
            sample_stat['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()

            sample_stats[validation_log_prob] = sample_stat

        experiment["sample_stats"] = sample_stats

        # Find best model, and train it on the joint train and validation set
        best_log_prob = np.min(list(sample_stats.keys()))
        best_perm = sample_stats[best_log_prob]

        logger.info(f"Retraining model with best permutation: {best_perm}, log_prob: {best_log_prob}")
        # -----------------------------------------------------------------------------
        # Train best model
        # -----------------------------------------------------------------------------
        start_time = time.time()
        try:
            model = model.condition_on(train_and_validate[inputs].values,
                                       train_and_validate[outputs].values[:, best_perm],
                                       keep_previous=False)
            model.fit_to_conditioning_data(optimizer_restarts=optimizer_restarts,
                                           optimizer=optimizer,
                                           trace=True)
        except Exception as e:
            logger.exception("Training failed: {}".format(str(e)))
            raise e

        experiment['train_time'] = time.time() - start_time

        # -----------------------------------------------------------------------------
        # Test best model
        # -----------------------------------------------------------------------------
        start_time = time.time()
        try:
            mean, _ = model.predict(test[inputs].values, numpy=True)
        except Exception as e:
            logger.exception("Prediction failed: {}, saving model!".format(str(e)))

            model.save("models/exceptions/" + experiment_file_name + "/model")
            raise e

        experiment['predict_time'] = time.time() - start_time

        diff = (test[outputs].values - mean)

        experiment['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
        experiment['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
        experiment['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()

        experiments.append(experiment)

        logger.info("Saving experiments to {}".format(experiment_file_path))
        with open(experiment_file_path, mode='w') as out_file:
            json.dump(experiments, out_file, sort_keys=True, indent=4)

    return experiments


def run_greedy_experiment(model,
                          data,
                          optimizer,
                          optimizer_restarts,
                          inputs,
                          outputs,
                          num_target_dims,
                          training_set_size,
                          validation_set_size,
                          test_set_size,
                          logdir,
                          matrix_factorized,
                          experiment_file_name,
                          seed,
                          rounds):
    experiment_file_path = os.path.join(logdir, experiment_file_name)

    # Make sure the directory exists
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    experiments = []

    # Set seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Split the data into training and validation set and the held-out test set
    train_and_validate, test = train_test_split(data,
                                                train_size=training_set_size + validation_set_size,
                                                test_size=test_set_size,
                                                random_state=seed)

    # =========================================================================
    # Perform Greedy search
    # =========================================================================

    for index in range(rounds):

        logger.info("-----------------------------------------------------------")
        logger.info(f"Training round: {index + 1}/{rounds} for training set size {training_set_size}, ")
        logger.info("-----------------------------------------------------------")

        experiment = {'index': index,
                      'size': training_set_size,
                      'inputs': inputs,
                      'outputs': outputs}

        if matrix_factorized:
            experiment["latent_size"] = model.latent_dim

        train, validate = train_test_split(data,
                                           train_size=training_set_size,
                                           test_size=validation_set_size,
                                           random_state=seed + index)

        start_time = time.time()
        try:
            model = model.condition_on(train[inputs].values, train[outputs].values[:, :], keep_previous=False)
            model.fit_greedy_ordering(train_xs=train[inputs].values,
                                      train_ys=train[outputs].values[:, :],
                                      validation_xs=validate[inputs].values,
                                      validation_ys=validate[outputs].values[:, :],
                                      trace=True,
                                      optimizer_restarts=optimizer_restarts,
                                      seed=seed + index,
                                      optimizer=optimizer,
                                      num_target_dimensions=num_target_dims)
        except Exception as e:
            logger.exception("Training failed: {}".format(str(e)))
            raise e

        experiment['train_time'] = time.time() - start_time

        save_path = f"{DEFAULT_MODEL_SAVE_DIR}/{experiment_file_name}/size_{training_set_size}/model_{index}/model"
        model.save(save_path)
        logger.info(f"Saved model to {save_path}!")

        start_time = time.time()

        try:
            mean, _ = model.predict(test[inputs].values, numpy=True)

        except Exception as e:
            logger.exception("Prediction failed: {}, saving model!".format(str(e)))

            model.save("models/exceptions/" + experiment_file_name + "/model")
            raise e
            # continue

        experiment['predict_time'] = time.time() - start_time

        diff = (test[outputs].values - mean)

        experiment["perm"] = [int(x) for x in model.permutation.numpy()]
        experiment['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
        experiment['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
        experiment['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()

        experiments.append(experiment)

        logger.info("Saving experiments to {}".format(experiment_file_path))
        with open(experiment_file_path, mode='w') as out_file:
            json.dump(experiments, out_file, sort_keys=True, indent=4)


def prepare_gpar_data(data, targets):
    output_labels = data.output_labels.copy()

    for target in targets:
        output_labels.remove(target)
        output_labels.append(target)

    return data.df, data.input_labels.copy(), output_labels


def main(args, seed=27, experiment_json_format="{}_train_{}_valid_{}_{}_experiments.json"):
    data = load_dataset(path=args.dataset, kind=args.task)

    model = None

    df, input_labels, output_labels = prepare_gpar_data(data, DATASET_TARGETS[args.task])

    if args.model == 'gpar':
        model = GPARModel(kernel=args.kernel,
                          input_dim=len(input_labels),
                          output_dim=len(output_labels),
                          initialization_heuristic=args.initialization,
                          verbose=args.verbose)

    elif args.model == 'mf-gpar':
        model = MatrixFactorizedGPARModel(kernel=args.kernel,
                                          input_dim=len(input_labels),
                                          output_dim=len(output_labels),
                                          latent_dim=args.latent_dim,
                                          initialization_heuristic=args.initialization,
                                          verbose=args.verbose)

    elif args.model == 'p-gpar':
        model = PermutedGPARModel(kernel=args.kernel,
                                  input_dim=len(input_labels),
                                  output_dim=len(output_labels),
                                  initialization_heuristic=args.initialization,
                                  verbose=args.verbose)

    # Perform experiments
    experiment_file_name = ""

    # If the model uses matrix factorization, then append the latent dimension to the file name
    if args.model in ["mf-gpar"]:
        experiment_file_name = experiment_json_format.format(f"{args.model}-{args.latent_dim}",
                                                             args.train_size,
                                                             args.validation_size,
                                                             args.search_mode)
    else:
        experiment_file_name = experiment_json_format.format(args.model,
                                                             args.train_size,
                                                             args.validation_size,
                                                             args.search_mode)

    if args.search_mode == "random_search":
        results = run_random_experiment(model=model,
                                        data=df,
                                        optimizer=args.optimizer,
                                        optimizer_restarts=args.num_optimizer_restarts,
                                        inputs=input_labels,
                                        outputs=output_labels,
                                        num_target_dims=args.num_target_dims,
                                        num_samples=args.num_samples,
                                        training_set_size=args.train_size,
                                        validation_set_size=args.validation_size,
                                        test_set_size=args.test_size,
                                        logdir=args.logdir,
                                        matrix_factorized=args.model == "mf-gpar",
                                        experiment_file_name=experiment_file_name,
                                        seed=seed,
                                        rounds=args.num_rounds,
                                        verbose=args.verbose)

    elif args.search_mode == "greedy_search":
        run_greedy_experiment(model=model,
                              data=df,
                              optimizer=args.optimizer,
                              optimizer_restarts=args.num_optimizer_restarts,
                              inputs=input_labels,
                              outputs=output_labels,
                              num_target_dims=args.num_target_dims,
                              training_set_size=args.train_size,
                              validation_set_size=args.validation_size,
                              test_set_size=args.test_size,
                              logdir=args.logdir,
                              matrix_factorized=args.model == "mf-gpar",
                              experiment_file_name=experiment_file_name,
                              seed=seed,
                              rounds=args.num_rounds)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-D', type=str, required=True, help="Path to the dataset.")

    parser.add_argument('--task',
                        '-T',
                        choices=AVAILABLE_DATASETS,
                        required=True,
                        help="Task for which we are providing the dataset.")

    parser.add_argument('--logdir',
                        type=str,
                        default="logs",
                        help="Path to the directory to which we will write the log files "
                             "for the experiment.")

    parser.add_argument(
        '--loglevel',
        choices=LOG_LEVELS,
    )

    parser.add_argument('--verbose', action="store_true", default=False, help="Turns on verbose logging")

    parser.add_argument("--num_rounds",
                        type=int,
                        default=5,
                        help="Number of rounds to perform for a single setting.")

    model_subparsers = parser.add_subparsers(title="model", dest="model", help="Model to fit to the data.")

    model_subparsers.required = True

    # =========================================================================
    # GPAR
    # =========================================================================

    gpar_mode = model_subparsers.add_parser("gpar",
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                            description="Use a GPAR model.")

    # =========================================================================
    # Matrix factorized GPAR
    # =========================================================================

    mf_gpar_mode = model_subparsers.add_parser("mf-gpar",
                                               formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                               description="use a GPAR Model with factorized length scale matrix")

    mf_gpar_mode.add_argument("--latent_dim", type=int, default=5, help="Effective dimension of the factorization.")

    # =========================================================================
    # Permuted GPAR
    # =========================================================================

    p_gpar_mode = model_subparsers.add_parser("p-gpar",
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                              description="use a Permuted GPAR Model")

    # Add common options to models
    for mode in [gpar_mode, mf_gpar_mode, p_gpar_mode]:
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

        experiment_subparsers = mode.add_subparsers(title="search_mode",
                                                    dest="search_mode",
                                                    help="Type of search to perform to find optimal ordering")

        experiment_subparsers.required = True

        # ---------------------------------------------------------------------
        # Random ordering search
        # ---------------------------------------------------------------------
        random_search_mode = experiment_subparsers.add_parser("random_search",
                                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                                              description="Random search")

        random_search_mode.add_argument("--num_samples",
                                        choices=AVAILABLE_RANDOM_SEARCH_OPTIONS,
                                        default="linear",
                                        help="Number of draws to perform as a function of the number of inputs.")

        # ---------------------------------------------------------------------
        # Greedy ordering search
        # ---------------------------------------------------------------------
        greedy_search_mode = experiment_subparsers.add_parser("greedy_search",
                                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                                              description="Greedy search")

        for search_mode in [random_search_mode, greedy_search_mode]:
            search_mode.add_argument("--train_size",
                                     type=int,
                                     default=100,
                                     help="Number of training examples to use.")

            search_mode.add_argument("--validation_size",
                                     type=int,
                                     default=50,
                                     help="Number of examples to use for validation.")

            search_mode.add_argument("--test_size",
                                     type=int,
                                     default=200,
                                     help="Number of examples to use for the held-out test set.")

            search_mode.add_argument("--num_target_dims",
                                     type=int,
                                     default=3,
                                     help="Number of target dimensions, "
                                          "for which we should not optimize the permutations")

    args = parser.parse_args()

    main(args)
