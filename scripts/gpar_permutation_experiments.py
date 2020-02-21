import logging
import argparse
import json
import time
import os
from typing import Sequence

from itertools import permutations

import numpy as np

from sklearn.model_selection import train_test_split

from boa.models.gpar import GPARModel
from boa.models.matrix_factorized_gpar import MatrixFactorizedGPARModel
from boa.models.gpar_perm import PermutedGPARModel
from boa.models.permutation_gp import PermutationGPModel

from boa.core import GaussianProcess, setup_logger
from boa.core.distribution import GumbelMatching

from boa.datasets.loader import load_dataset

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

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
        train, validate = train_test_split(train_and_validate,
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
        best_perm = sample_stats[best_log_prob]["perm"]

        experiment["perm"] = best_perm

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

        train, validate = train_test_split(train_and_validate,
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


def run_hierarchical_bayesopt_experiment(model,
                                         data,
                                         optimizer,
                                         optimizer_restarts,
                                         inputs,
                                         outputs,
                                         num_target_dims,
                                         training_set_size,
                                         validation_set_size,
                                         test_set_size,
                                         num_warmup_samples,
                                         num_bayesopt_steps,
                                         num_bayesopt_candidates,
                                         median_heuristic_only,
                                         logdir,
                                         matrix_factorized,
                                         experiment_file_name,
                                         seed,
                                         rounds,
                                         xi=0.01):
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

    num_permuted_dimensions = len(outputs) - num_target_dims

    if num_permuted_dimensions > 10:
        raise Exception("Too many dimensions to optimize, this is not implemented yet!")

    # Create uniform distribution over permutations
    uniform_gm = GumbelMatching(weight_matrix=tf.zeros((num_permuted_dimensions,
                                                        num_permuted_dimensions),
                                                       dtype=tf.float64))

    # =========================================================================
    # Perform hierarchical BayesOpt
    # =========================================================================
    logger.info(f"Performing hieararchical BayesOpt order search!")

    # We "restart" the search rounds number of times
    for index in range(rounds):

        experiment = {'index': index,
                      'size': training_set_size,
                      'inputs': inputs,
                      'outputs': outputs}

        # In each round we do a different train-validate split
        train, validate = train_test_split(train_and_validate,
                                           train_size=training_set_size,
                                           test_size=validation_set_size,
                                           random_state=seed + index)

        # ---------------------------------------------------------------------
        # Step 1: Perform a few random search steps so that we can train the
        # hyperparameters of the surrogate model
        # ---------------------------------------------------------------------
        warmup_stats = {}

        for sample_number in range(num_warmup_samples):

            # Set the seed for reproducibility in a way that we never use the same seed
            tf.random.set_seed(seed * (index + 1) + sample_number)

            # Draw a new permutation
            perm = uniform_gm.sample(as_tuple=True)

            # convert numpy.int64 to python int
            perm = tuple([p.item() for p in perm])

            # Complete the permutation to all inputs
            perm = perm + tuple(range(num_permuted_dimensions, len(outputs)))

            warmup_stat = {"perm": perm}

            logger.info(f"Warmup training round: {index + 1}/{rounds} for training set size {training_set_size}, "
                        f"permutation #{sample_number + 1} out of {num_warmup_samples}: {perm}")

            if matrix_factorized:
                warmup_stat["latent_size"] = model.latent_dim

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

            warmup_stat['train_time'] = time.time() - start_time

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

            warmup_stat['predict_time'] = time.time() - start_time

            diff = (validate[outputs].values - mean)

            warmup_stat['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
            warmup_stat['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
            warmup_stat['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()

            warmup_stats[validation_log_prob] = warmup_stat

        # For testing only
        #warmup_stats = {-855.8674197639222: {'perm': (1, 0, 6, 5, 4, 7, 3, 8, 2, 9, 10, 11), 'train_time': 139.47014737129211, 'predict_time': 0.38406872749328613, 'mean_abs_err': [1.055083605426268, 1.0821624048065686, 83570.14800597663, 25.680902306994085, 7.680843002897069, 882642.8720624018, 83542.7856493097, 871615.2946727732, 75.44358542606287, 8.28024456248029, 25152.73502910184, 16215.967169160449], 'mean_squ_err': [1.997591251458251, 1.5726827769872294, 7067348835.387845, 1723.3486801505846, 172.25821064964074, 1142381188399.5256, 7061934677.559448, 1134757632206.0813, 5714.6874589591525, 199.503050122186, 1075081351.0003712, 950924834.4065049], 'rmse': [1.4133616845868757, 1.2540664962382295, 84067.52545060337, 41.513234999823666, 13.124717545518484, 1068822.3371540874, 84035.31803687928, 1065250.032718179, 75.59555184638282, 14.124554864567804, 32788.433189165524, 30837.069160452083]}, -676.2210324953625: {'perm': (6, 8, 3, 4, 7, 1, 5, 0, 2, 9, 10, 11), 'train_time': 149.60682010650635, 'predict_time': 0.3812880516052246, 'mean_abs_err': [83549.87957578048, 73.5734303262038, 25.641864107280544, 12.90513203163348, 871008.0335877128, 2.7035541302120585, 83565.23222205912, 871688.5715432717, 75.44326384696026, 11.963277070891152, 41132.94635963661, 12474.749111234361], 'mean_squ_err': [7065679606.543886, 5429.989144776781, 1223.5052932582198, 362.4137026413832, 1135375808661.709, 11.31246435889567, 7065725177.989665, 1134885487147.7039, 5714.642835668615, 287.7460722411604, 2739613566.215577, 414171663.80137956], 'rmse': [84057.59695913205, 73.68846005160361, 34.97864052901742, 19.03716635010009, 1065540.1487798144, 3.3634007133994115, 84057.86803143217, 1065310.0427329612, 75.59525670085799, 16.96307968032811, 52341.31796406714, 20351.20791995845]}}

        experiment["warmup_stats"] = warmup_stats

        print("Warmup stats:")
        print(warmup_stats)

        # ---------------------------------------------------------------------
        # Step 2: Create surrogate model and train its hyperparameters using
        # the data we just collected
        # ---------------------------------------------------------------------
        train_permutations = []
        train_log_probs = []

        for log_prob, warmup_stat in warmup_stats.items():
            train_log_probs.append(log_prob)

            # Only use the non-target dimensions
            train_permutations.append(warmup_stat['perm'][:-num_target_dims])

        perm_gp = PermutationGPModel(kernel='tau_rbf',
                                     input_dim=num_permuted_dimensions)

        logger.info("Fitting hyperparameters of the permutation surrogate model")
        perm_gp = perm_gp.condition_on(train_permutations,
                                       train_log_probs)

        try:
            perm_gp.fit_to_conditioning_data(optimizer_restarts=optimizer_restarts,
                                             optimizer='l-bfgs-b',
                                             trace=True,
                                             err_level='raise',
                                             median_heuristic_only=median_heuristic_only)

        except Exception as e:
            logger.exception("Training the surrogate model failed: {}".format(str(e)))
            raise e

        # --------------------------------------------------------------------
        # Step 3: Use the trained model to perform BayesOpt by using
        # Expected Improvement as the acquisition function
        # --------------------------------------------------------------------

        # Generate all possible permutations
        # candidate_perms = list(permutations(range(num_permuted_dimensions)))[:100]

        observed_perms = train_permutations
        observed_log_probs = train_log_probs

        standard_normal = tfd.Normal(loc=tf.cast(0., tf.float64), scale=tf.cast(1., tf.float64))

        bayesopt_stats = {}

        xi = tf.cast(xi, tf.float64)

        def generate_candidates(num_candidates):
            candidates = []

            s = 0
            while s < num_candidates:
                samp = uniform_gm.sample(as_tuple=True)

                if samp in observed_perms:
                    continue

                candidates.append(samp)
                s = s + 1

            return candidates

        for sample_number in range(num_bayesopt_steps):

            # Generate 10000 candidate permutations
            candidate_perms = generate_candidates(num_bayesopt_candidates)

            # Calculate statistics for candidate permutations
            logger.info("Calculating statistics for candidate permutations")
            perm_means, perm_vars = perm_gp.predict(candidate_perms)
            perm_stds = tf.math.sqrt(perm_vars)

            logger.info("Calculcating EI acquisition function")
            max_observed_log_prob = tf.reduce_max(observed_log_probs)
            max_observed_log_prob = tf.cast(max_observed_log_prob, tf.float64)

            z = perm_means - max_observed_log_prob - xi

            z_ = tf.where(perm_vars > 1e-10, z / perm_stds, 0.)

            expected_imp = z * standard_normal.cdf(z_) + perm_stds * standard_normal.prob(z_)
            expected_imp = tf.where(perm_vars > 1e-10, expected_imp, 0.)

            next_evaluation_point = candidate_perms[tf.argmax(expected_imp).numpy()[0]]

            # extend the permutation to include the target dimensions
            extended_next_evaluation_point = next_evaluation_point + tuple(range(num_permuted_dimensions, len(outputs)))

            bayesopt_stat = {"perm": extended_next_evaluation_point}

            logger.info(f"BayesOpt training round: {index + 1}/{rounds} for training set size {training_set_size}, "
                        f"permutation #{sample_number + 1} out of {num_bayesopt_steps}: {next_evaluation_point}")

            if matrix_factorized:
                bayesopt_stat["latent_size"] = model.latent_dim

            # -----------------------------------------------------------------------------
            # Train model
            # -----------------------------------------------------------------------------
            start_time = time.time()
            try:
                model = model.condition_on(train[inputs].values,
                                           train[outputs].values[:, extended_next_evaluation_point],
                                           keep_previous=False)
                model.fit_to_conditioning_data(optimizer_restarts=optimizer_restarts,
                                               optimizer=optimizer,
                                               trace=True)
            except Exception as e:
                logger.exception("Training failed: {}".format(str(e)))
                raise e

            bayesopt_stat['train_time'] = time.time() - start_time

            # -----------------------------------------------------------------------------
            # Validate model
            # -----------------------------------------------------------------------------
            start_time = time.time()
            try:
                mean, _ = model.predict(validate[inputs].values, numpy=True)

                validation_log_prob = model.log_prob(validate[inputs].values,
                                                     validate[outputs].values[:, extended_next_evaluation_point],
                                                     use_conditioning_data=True,
                                                     numpy=True)

            except Exception as e:
                logger.exception("Prediction failed: {}, saving model!".format(str(e)))

                model.save("models/exceptions/" + experiment_file_name + "/model")
                raise e

            bayesopt_stat['predict_time'] = time.time() - start_time

            diff = (validate[outputs].values - mean)

            bayesopt_stat['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
            bayesopt_stat['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
            bayesopt_stat['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()

            bayesopt_stats["validation_log_prob"] = validation_log_prob

            # Append the observed stuff to our list
            observed_perms.append(next_evaluation_point)
            observed_log_probs.append(validation_log_prob)

            # -----------------------------------------------------------------------------
            # Re-train model hyperparameters on newly observed data
            # -----------------------------------------------------------------------------
            logger.info("Retraining GP with newly observed data!")
            perm_gp = perm_gp.condition_on(observed_perms, observed_log_probs, keep_previous=False)
            perm_gp.fit_to_conditioning_data(optimizer_restarts=optimizer_restarts,
                                             optimizer='l-bfgs-b',
                                             trace=True,
                                             median_heuristic_only=median_heuristic_only)

        experiment["bayesopt_stats"] = bayesopt_stats

        # The best permutation is the one with the highest log probability
        best_perm = observed_perms[tf.argmax(observed_log_probs).numpy()]
        best_perm = best_perm + tuple(range(num_permuted_dimensions, len(outputs)))

        best_log_prob = tf.reduce_max(observed_log_probs)

        experiment["perm"] = best_perm

        # ---------------------------------------------------------------------
        # Step 4: Retrain the model on the joint training-validation set
        # ---------------------------------------------------------------------
        logger.info(f"Retraining model with best permutation: {best_perm}, log_prob: {best_log_prob}")
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

    elif args.search_mode == "hbo":
        run_hierarchical_bayesopt_experiment(model=model,
                                             data=df,
                                             optimizer=args.optimizer,
                                             optimizer_restarts=args.num_optimizer_restarts,
                                             inputs=input_labels,
                                             outputs=output_labels,
                                             num_target_dims=args.num_target_dims,
                                             training_set_size=args.train_size,
                                             validation_set_size=args.validation_size,
                                             test_set_size=args.test_size,
                                             num_warmup_samples=args.num_warmup_samples,
                                             num_bayesopt_steps=args.num_bayesopt_steps,
                                             num_bayesopt_candidates=args.num_bayesopt_candidates,
                                             median_heuristic_only=args.median_heuristic_only,
                                             logdir=args.logdir,
                                             matrix_factorized=args.model == "mf-gpar",
                                             experiment_file_name=experiment_file_name,
                                             seed=seed,
                                             rounds=args.num_rounds,
                                             xi=args.xi)

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

        # ---------------------------------------------------------------------
        # Hierarchical BayesOpt search
        # ---------------------------------------------------------------------
        hier_bayesopt_search_mode = experiment_subparsers.add_parser("hbo",
                                                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                                                     description="Hierarchical Bayesian Optimization")

        hier_bayesopt_search_mode.add_argument("--num_warmup_samples",
                                               type=int,
                                               default=5,
                                               help="Number of random permutations to draw to train surrogate model "
                                                    "first")
        hier_bayesopt_search_mode.add_argument("--num_bayesopt_steps",
                                               type=int,
                                               default=10,
                                               help="Number of BayesOpt steps to perform to find the best permutation")
        hier_bayesopt_search_mode.add_argument("--xi",
                                               type=float,
                                               default=0.01,
                                               help="Exploration constant for the Expected Imporvement acqusition "
                                                    "function.")
        hier_bayesopt_search_mode.add_argument("--median_heuristic_only",
                                               action='store_true',
                                               default=False,
                                               help="Whether we should only set the length scales to the median"
                                                    "of the permutation distances instead of fitting.")

        hier_bayesopt_search_mode.add_argument("--num_bayesopt_candidates",
                                               type=int,
                                               default=10000,
                                               help="Number of candidate permutations to be generated.")

        for search_mode in [random_search_mode, greedy_search_mode, hier_bayesopt_search_mode]:
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
