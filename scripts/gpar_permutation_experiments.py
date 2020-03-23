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

from boa import ROOT_DIR
from boa.core import GaussianProcess, setup_logger
from boa.core.distribution import GumbelMatching

import tensorflow as tf
import tensorflow_probability as tfp

import datetime
from sacred import Experiment

from dataset_config import dataset_ingredient, load_dataset, prepare_gpar_data

ex = Experiment("gpar_permutation_experiment", ingredients=[dataset_ingredient])

tfd = tfp.distributions


@ex.config
def experiment_config(dataset):
    task = "fft"
    model = "gpar"

    verbose = True

    # Search to use to find optimal permutation
    search_mode = "hbo"

    # Number of training examples to use
    train_size = 100

    # Number of examples to use for validation.
    validation_size = 50

    # Number of examples to use for the held-out test set.
    test_size = 200

    # Number of target dimensions, for which we should not optimize the permutations
    num_target_dims = 3

    if search_mode == "random":
        num_samples = 35

    elif search_mode == "greedy":
        pass

    elif search_mode == "hbo":
        # Number of random permutations to draw to train surrogate model
        num_warmup_samples = 10

        # Number of BayesOpt steps to perform to find the best permutation
        num_bayesopt_steps = 25

        # Exploration constant for the Expected Imporvement acqusition
        xi = 0.01

        # Number of candidate permutations to be generated.
        num_bayesopt_candidates = 10000

        # Whether we should only set the length scales to the median of the permutation distances instead of fitting.
        median_heuristic_only = False

        # Permutation distance to use in the modified GP kernel
        distance_kind = "inverse_weighted_kendall"

    # Number of experiments to perform
    rounds = 5

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Path to the directory to which we will write the log files
    log_dir = f"{ROOT_DIR}/../logs/{task}/ordering/"
    save_dir = f"{ROOT_DIR}/../models/{task}/ordering/{current_time}/"

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

    if model == "gpar":
        log_dir = f"{log_dir}/{search_mode}/{model}/{current_time}/"
        log_path = f"{log_dir}/train_{train_size}_valid_" \
                   f"{validation_size}_{(distance_kind + '_') if search_mode == 'hbo' else ''}_" \
                   f"{'median_only_' if median_heuristic_only else ''}experiments.json"

    elif model == "mf_gpar":
        # Effective dimension of the factorization.
        latent_dim = 5
        matrix_factorized = True
        log_dir = f"{log_dir}/{search_mode}/{model}-{latent_dim}/{current_time}/"
        log_path = f"{log_dir}/train_{train_size}_valid_" \
                   f"{validation_size}{('_' + distance_kind) if search_mode == 'hbo' else ''}" \
                   f"{'_median_only' if median_heuristic_only else ''}_experiments.json"


AVAILABLE_DATASETS = ["fft", "stencil3d", "gemm"]
AVAILABLE_OPTIMIZERS = ["l-bfgs-b", "adam"]
AVAILABLE_INITIALIZATION = ["median", "random", "dim_median"]

AVAILABLE_RANDOM_SEARCH_OPTIONS = ["linear", "log-linear", "quadratic"]

AVAILABLE_HBO_DISTANCE_KINDS = ["kendall", "inverse_weighted_kendall", "weighted_kendall", "spearman"]

# Set CPU as available physical device
tf.config.experimental.set_visible_devices([], 'GPU')


def generate_dist_1_neighbourhood(perm):
    neighbourhood = []

    for i in range(len(perm) - 1):
        new_perm = list(perm)
        new_perm[i:i + 2] = new_perm[i:i + 2][::-1]

        neighbourhood.append(tuple(new_perm))

    return neighbourhood


def generate_neighbourhood(perm, dist=1):
    neighbourhood = {tuple(perm)}

    for d in range(dist):
        n_p = set([])
        for p in neighbourhood:
            n_p.update(generate_dist_1_neighbourhood(p))

        neighbourhood.update(n_p)

    return list(neighbourhood)


@ex.capture
def run_random_experiment(model,
                          data,
                          inputs: Sequence[str],
                          outputs: Sequence[str],

                          optimizer,
                          optimizer_restarts,
                          num_target_dims,
                          train_size,
                          validation_size,
                          test_size,
                          log_dir,
                          log_path,
                          matrix_factorized,
                          num_samples,
                          rounds,
                          _seed,
                          _log):
    # Make sure the directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    experiments = []

    # Set seed for reproducibility
    np.random.seed(_seed)
    tf.random.set_seed(_seed)

    # Split the data into training and validation set and the held-out test set
    train_and_validate, test = train_test_split(data,
                                                train_size=train_size + validation_size,
                                                test_size=test_size,
                                                random_state=_seed)

    # =========================================================================
    # Perform random search
    # =========================================================================
    num_permuted_dimensions = len(outputs) - num_target_dims

    # Create uniform distribution over permutations
    uniform_gm = GumbelMatching(weight_matrix=tf.zeros((num_permuted_dimensions,
                                                        num_permuted_dimensions),
                                                       dtype=tf.float64))

    _log.info(f"Performing random order search using {num_samples} samples!")

    # We "restart" the search rounds number of times
    for index in range(rounds):

        experiment = {'index': index,
                      'size': train_size,
                      'inputs': inputs,
                      'outputs': outputs}

        # In each round we do a different train-validate split
        train, validate = train_test_split(train_and_validate,
                                           train_size=train_size,
                                           test_size=validation_size,
                                           random_state=_seed + index)

        # We will record the statistics for every permutation selected, and choose the best one of these
        sample_stats = {}

        for sample_number in range(num_samples):

            # Set the seed for reproducibility in a way that we never use the same seed
            tf.random.set_seed(_seed * (index + 1) + sample_number)

            # Draw a new permutation
            perm = uniform_gm.sample(as_tuple=True)

            # convert numpy.int64 to python int
            perm = tuple([p.item() for p in perm])

            # Complete the permutation to all inputs
            perm = perm + tuple(range(num_permuted_dimensions, len(outputs)))

            sample_stat = {"perm": perm}

            _log.info(f"Training round: {index + 1}/{rounds} for training set size {train_size}, "
                      f"permutation #{sample_number + 1} out of {num_samples}: {perm}")

            if matrix_factorized:
                sample_stat["latent_size"] = model.latent_dim

            # -----------------------------------------------------------------------------
            # Train model
            # -----------------------------------------------------------------------------
            start_time = time.time()
            try:
                model = model.condition_on(train[inputs].values, train[outputs].values[:, perm], keep_previous=False)
                model.fit(optimizer_restarts=optimizer_restarts, optimizer=optimizer, trace=True)
            except Exception as e:
                _log.exception("Training failed: {}".format(str(e)))
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
                _log.exception("Prediction failed: {}, saving model!".format(str(e)))
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

        _log.info(f"Retraining model with best permutation: {best_perm}, log_prob: {best_log_prob}")
        # -----------------------------------------------------------------------------
        # Train best model
        # -----------------------------------------------------------------------------
        start_time = time.time()
        try:
            model = model.condition_on(train_and_validate[inputs].values,
                                       train_and_validate[outputs].values[:, best_perm],
                                       keep_previous=False)
            model.fit(optimizer_restarts=optimizer_restarts,
                                           optimizer=optimizer,
                                           trace=True)
        except Exception as e:
            _log.exception("Training failed: {}".format(str(e)))
            raise e

        experiment['train_time'] = time.time() - start_time

        # -----------------------------------------------------------------------------
        # Test best model
        # -----------------------------------------------------------------------------
        start_time = time.time()
        try:
            mean, _ = model.predict(test[inputs].values, numpy=True)
        except Exception as e:
            _log.exception("Prediction failed: {}, saving model!".format(str(e)))
            raise e

        experiment['predict_time'] = time.time() - start_time

        diff = (test[outputs].values - mean)

        experiment['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
        experiment['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
        experiment['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()

        experiments.append(experiment)

        _log.info(f"Saving experiments to {log_path}")
        with open(log_path, mode='w') as out_file:
            json.dump(experiments, out_file, sort_keys=True, indent=4)

    return experiments


@ex.capture
def run_greedy_experiment(model,
                          data,
                          inputs,
                          outputs,

                          optimizer,
                          optimizer_restarts,
                          num_target_dims,
                          train_size,
                          validation_size,
                          test_size,
                          log_dir,
                          log_path,
                          save_dir,
                          matrix_factorized,
                          rounds,
                          _seed,
                          _log):
    # Make sure the directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    experiments = []

    # Set seed for reproducibility
    np.random.seed(_seed)
    tf.random.set_seed(_seed)

    # Split the data into training and validation set and the held-out test set
    train_and_validate, test = train_test_split(data,
                                                train_size=train_size + validation_size,
                                                test_size=test_size,
                                                random_state=_seed)

    # =========================================================================
    # Perform Greedy search
    # =========================================================================

    for index in range(rounds):

        _log.info("-----------------------------------------------------------")
        _log.info(f"Training round: {index + 1}/{rounds} for training set size {train_size}, ")
        _log.info("-----------------------------------------------------------")

        experiment = {'index': index,
                      'size': train_size,
                      'inputs': inputs,
                      'outputs': outputs}

        if matrix_factorized:
            experiment["latent_size"] = model.latent_dim

        train, validate = train_test_split(train_and_validate,
                                           train_size=train_size,
                                           test_size=validation_size,
                                           random_state=_seed + index)

        start_time = time.time()
        try:
            model = model.condition_on(train[inputs].values, train[outputs].values[:, :], keep_previous=False)
            model.fit_greedy_ordering(train_xs=train[inputs].values,
                                      train_ys=train[outputs].values[:, :],
                                      validation_xs=validate[inputs].values,
                                      validation_ys=validate[outputs].values[:, :],
                                      trace=True,
                                      optimizer_restarts=optimizer_restarts,
                                      seed=_seed + index,
                                      optimizer=optimizer,
                                      num_target_dimensions=num_target_dims)
        except Exception as e:
            _log.exception("Training failed: {}".format(str(e)))
            raise e

        experiment['train_time'] = time.time() - start_time

        save_path = f"{save_dir}/size_{train_size}/model_{index}/model"
        model.save(save_path, )
        _log.info(f"Saved model to {save_path}!")

        start_time = time.time()

        try:
            mean, _ = model.predict(test[inputs].values, numpy=True)

        except Exception as e:
            raise e

        experiment['predict_time'] = time.time() - start_time

        diff = (test[outputs].values - mean)

        experiment["perm"] = [int(x) for x in model.permutation.numpy()]
        experiment['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
        experiment['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
        experiment['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()

        experiments.append(experiment)

        _log.info("Saving experiments to {}".format(log_path))
        with open(log_path, mode='w') as out_file:
            json.dump(experiments, out_file, sort_keys=True, indent=4)


@ex.capture
def run_hierarchical_bayesopt_experiment(model,
                                         data,
                                         inputs,
                                         outputs,

                                         optimizer,
                                         optimizer_restarts,
                                         num_target_dims,
                                         train_size,
                                         validation_size,
                                         test_size,
                                         num_warmup_samples,
                                         num_bayesopt_steps,
                                         num_bayesopt_candidates,
                                         median_heuristic_only,
                                         log_dir,
                                         log_path,
                                         save_dir,
                                         matrix_factorized,
                                         rounds,
                                         distance_kind,
                                         xi,
                                         _seed,
                                         _log):
    # Make sure the directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    experiments = []

    # Set seed for reproducibility
    np.random.seed(_seed)
    tf.random.set_seed(_seed)

    # Split the data into training and validation set and the held-out test set
    train_and_validate, test = train_test_split(data,
                                                train_size=train_size + validation_size,
                                                test_size=test_size,
                                                random_state=_seed)

    num_permuted_dimensions = len(outputs) - num_target_dims

    target_dims = tuple(range(num_permuted_dimensions, len(outputs)))

    if num_permuted_dimensions > 10:
        raise Exception("Too many dimensions to optimize, this is not implemented yet!")

    # Create uniform distribution over permutations
    uniform_gm = GumbelMatching(weight_matrix=tf.zeros((num_permuted_dimensions,
                                                        num_permuted_dimensions),
                                                       dtype=tf.float64))

    # =========================================================================
    # Perform hierarchical BayesOpt
    # =========================================================================
    _log.info(f"Performing hieararchical BayesOpt order search!")

    # We "restart" the search rounds number of times
    for index in range(rounds):

        experiment = {'index': index,
                      'size': train_size,
                      'inputs': inputs,
                      'outputs': outputs}

        # In each round we do a different train-validate split
        train, validate = train_test_split(train_and_validate,
                                           train_size=train_size,
                                           test_size=validation_size,
                                           random_state=_seed + index)

        # ---------------------------------------------------------------------
        # Step 1: Perform a few random search steps so that we can train the
        # hyperparameters of the surrogate model
        # ---------------------------------------------------------------------
        warmup_stats = {}

        for sample_number in range(num_warmup_samples):

            # Set the seed for reproducibility in a way that we never use the same seed
            tf.random.set_seed(_seed * (index + 1) + sample_number)

            # Draw a new permutation
            perm = uniform_gm.sample(as_tuple=True)

            # convert numpy.int64 to python int
            perm = tuple([p.item() for p in perm])

            # Complete the permutation to all inputs
            perm = perm + target_dims

            warmup_stat = {"perm": perm}

            _log.info(f"Warmup training round: {index + 1}/{rounds} for training set size {train_size}, "
                      f"permutation #{sample_number + 1} out of {num_warmup_samples}: {perm}")

            if matrix_factorized:
                warmup_stat["latent_size"] = model.latent_dim

            # -----------------------------------------------------------------------------
            # Train model
            # -----------------------------------------------------------------------------
            start_time = time.time()
            try:
                model = model.condition_on(train[inputs].values, train[outputs].values[:, perm], keep_previous=False)
                model.fit(optimizer_restarts=optimizer_restarts, optimizer=optimizer, trace=True)
            except Exception as e:
                _log.exception("Training failed: {}".format(str(e)))
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
                                                     target_dims=target_dims,
                                                     use_conditioning_data=True,
                                                     numpy=True)

            except Exception as e:
                raise e

            warmup_stat['predict_time'] = time.time() - start_time

            diff = (validate[outputs].values - mean)

            warmup_stat['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
            warmup_stat['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
            warmup_stat['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()

            warmup_stats[validation_log_prob] = warmup_stat

        # For testing only
        # Full log liks
        # warmup_stats = {'4292.187538379905': {'mean_abs_err': [1.032979288561258, 0.860242457989242, 3.510043521252998, 6.410245121588141, 5.369068165165629, 969480.8252440335, 84545.25836285923, 884923.1154522442, 75.51619136189562, 4.5746483024400995, 13598.324993567998, 588.0510733213648], 'mean_squ_err': [1.386600530279279, 0.9087905659460981, 20.532792457499955, 92.05394785902189, 72.78155996741175, 1422203235662.1047, 7233966937.994184, 1265980626171.4304, 5717.759288025154, 60.03447777895762, 387570278.4985512, 1213837.4110947177], 'perm': [1, 0, 5, 4, 3, 7, 8, 6, 2, 9, 10, 11], 'predict_time': 0.44308662414550704, 'rmse': [1.177540033408325, 0.953305074960843, 4.531312443155951, 9.594474861034442, 8.531210932066546, 1192561.6276159922, 85052.73033826829, 1125158.044974763, 75.61586664202926, 7.748191903854577, 19686.804679748086, 1101.7428970021624], 'train_time': 495.51788330078125}, '4549.329503583338': {'mean_abs_err': [1.032979097473507, 0.860242671331473, 84619.75237433052, 27.989350961613635, 4.162202498567493, 969456.714823122, 84589.97198426673, 969472.535422948, 75.51626400006728, 4.498776722571448, 13598.776408441941, 622.2617196677124], 'mean_squ_err': [1.386600126620054, 0.908791117000822, 7245487409.640569, 1671.9544599575402, 54.66747891465511, 1422138839492.3462, 7241322720.034418, 1422586086332.6917, 5717.770727411165, 60.654726229884034, 387487347.04877365, 1378035.9182329848], 'perm': [1, 0, 6, 5, 4, 7, 3, 8, 2, 9, 10, 11], 'predict_time': 0.45433759689331005, 'rmse': [1.1775398620089481, 0.9533053639840811, 85120.42886193989, 40.88953973765834, 7.393745932519937, 1192534.628215192, 85095.96183153709, 1192722.1329097115, 75.61594228343098, 7.788114420697993, 19684.69829712342, 1173.8977460720268], 'train_time': 558.2354316711426}, '4637.00880712224': {'mean_abs_err': [84619.59245854903, 73.65174763445371, 31.701897748881716, 6.634369817641792, 969465.9619268243, 2.8003220430731233, 84617.2840568614, 969545.2121621674, 75.5162330943744, 5.180872011696941, 12661.844268780937, 567.1756503890641], 'mean_squ_err': [7245349339.470062, 5437.621663839473, 1783.1009519511472, 124.50502352898125, 1422342821400.0085, 12.055479530916108, 7246122786.91489, 1422727135956.4111, 5717.766826033363, 93.51681136127381, 358592664.3801566, 671549.2289316106], 'perm': [6, 8, 3, 4, 7, 1, 5, 0, 2, 9, 10, 11], 'predict_time': 0.44295501708984303, 'rmse': [85119.6178296758, 73.74023097224115, 42.226780032950025, 11.158181909656307, 1192620.1496704677, 3.472100161417597, 85124.16100564452, 1192781.260733254, 75.61591648610339, 9.670409058632101, 18936.543094771987, 819.4810729550809], 'train_time': 610.1383054256439}, '4680.804428823511': {'mean_abs_err': [2.421467679043514, 25.839244884345526, 1.973530382868172, 31.49921768314671, 24.21610626868137, 28.619111945465537, 884953.3424672825, 969472.4740868494, 84555.17332894664, 5.5238588446235966, 14819.758128589076, 626.9281616917951], 'mean_squ_err': [9.590976608854447, 1306.392234186631, 6.076258900637832, 1998.2744383191166, 1431.0613524275877, 1507.742943197166, 1265583314917.1582, 1422586075728.8247, 7234878248.2927475, 107.18616189177204, 577924341.935567, 1057700.3787724837], 'perm': [5, 4, 1, 2, 0, 3, 7, 8, 6, 9, 10, 11], 'predict_time': 0.47697186470031705, 'rmse': [3.096930191149688, 36.14404839232361, 2.465006876387535, 44.70206302083962, 37.829371557396875, 38.82966576210985, 1124981.4731439617, 1192722.1284644739, 85058.08749491577, 10.353074996916233, 24040.05702854232, 1028.4456129385178], 'train_time': 470.9649953842163}, '4717.454930768087': {'mean_abs_err': [2.786588541991134, 969524.5419655278, 27.350136433732523, 28.668153859458815, 84594.35376517131, 2.791137651673889, 84545.26926495423, 969544.5360175238, 47.07861789121798, 4.812456695535601, 12197.756551516153, 570.3133328382962], 'mean_squ_err': [10.36246487954847, 1422444163107.2485, 1422.1240673314283, 1781.22269029514, 7240898663.690995, 12.04491927085999, 7233973971.79609, 1422722164517.6147, 2722.685052019315, 86.99341365573791, 376317360.1468299, 723758.9497507092], 'perm': [2, 7, 4, 0, 6, 1, 8, 5, 3, 9, 10, 11], 'predict_time': 0.44226384162902804, 'rmse': [3.21907826552081, 1192662.6359148042, 37.7110602785367, 42.204534001634705, 85093.47015894343, 3.470579097335197, 85052.77168791203, 1192779.1767622433, 52.179354653151044, 9.327025981294247, 19398.90100358342, 850.7402363534413], 'train_time': 494.60453748703003}}

        # Target log liks
        # warmup_stats = {1385.4214322629086: {'perm': (1, 0, 6, 5, 4, 7, 3, 8, 2, 9, 10, 11), 'train_time': 549.5277528762817, 'predict_time': 0.31821537017822266, 'mean_abs_err': [1.0329790974735074, 0.8602426713314734, 84619.75237433052, 27.989350961613635, 4.162202498567493, 969456.714823122, 84589.97198426673, 969472.535422948, 75.51626400006728, 4.498776722571448, 13598.776408441941, 622.2617196677124], 'mean_squ_err': [1.386600126620054, 0.9087911170008225, 7245487409.640569, 1671.9544599575402, 54.66747891465511, 1422138839492.3462, 7241322720.034418, 1422586086332.6917, 5717.770727411165, 60.654726229884034, 387487347.04877365, 1378035.9182329848], 'rmse': [1.1775398620089488, 0.9533053639840817, 85120.42886193989, 40.88953973765834, 7.393745932519937, 1192534.628215192, 85095.96183153709, 1192722.1329097115, 75.61594228343098, 7.788114420697993, 19684.69829712342, 1173.8977460720268]}, 1385.421437771198: {'perm': (6, 8, 3, 4, 7, 1, 5, 0, 2, 9, 10, 11), 'train_time': 596.0647881031036, 'predict_time': 0.2819077968597412, 'mean_abs_err': [84619.59245854903, 73.65174763445371, 31.701897748881716, 6.634369817641792, 969465.9619268243, 2.800322043073123, 84617.2840568614, 969545.2121621674, 75.5162330943744, 5.180872011696941, 12661.844268780937, 567.1756503890641], 'mean_squ_err': [7245349339.470062, 5437.621663839473, 1783.1009519511472, 124.50502352898125, 1422342821400.0085, 12.055479530916108, 7246122786.91489, 1422727135956.4111, 5717.766826033363, 93.51681136127381, 358592664.3801566, 671549.2289316106], 'rmse': [85119.6178296758, 73.74023097224115, 42.226780032950025, 11.158181909656307, 1192620.1496704677, 3.4721001614175977, 85124.16100564452, 1192781.260733254, 75.61591648610339, 9.670409058632101, 18936.543094771987, 819.4810729550809]}, 1385.4214339497223: {'perm': (5, 4, 1, 2, 0, 3, 7, 8, 6, 9, 10, 11), 'train_time': 455.1098527908325, 'predict_time': 0.2822575569152832, 'mean_abs_err': [2.4214676790435146, 25.839244884345526, 1.973530382868172, 31.49921768314671, 24.21610626868137, 28.619111945465537, 884953.3424672825, 969472.4740868494, 84555.17332894664, 5.523858844623597, 14819.758128589076, 626.9281616917951], 'mean_squ_err': [9.590976608854447, 1306.392234186631, 6.076258900637832, 1998.2744383191166, 1431.0613524275877, 1507.742943197166, 1265583314917.1582, 1422586075728.8247, 7234878248.2927475, 107.18616189177204, 577924341.935567, 1057700.3787724837], 'rmse': [3.0969301911496885, 36.14404839232361, 2.465006876387535, 44.70206302083962, 37.829371557396875, 38.82966576210985, 1124981.4731439617, 1192722.1284644739, 85058.08749491577, 10.353074996916233, 24040.05702854232, 1028.4456129385178]}, 1385.4214360352953: {'perm': (1, 0, 5, 4, 3, 7, 8, 6, 2, 9, 10, 11), 'train_time': 476.7164604663849, 'predict_time': 0.2901787757873535, 'mean_abs_err': [1.0329792885612585, 0.860242457989242, 3.510043521252998, 6.410245121588141, 5.369068165165629, 969480.8252440335, 84545.25836285923, 884923.1154522442, 75.51619136189562, 4.574648302440099, 13598.324993567998, 588.0510733213648], 'mean_squ_err': [1.3866005302792794, 0.9087905659460985, 20.532792457499955, 92.05394785902189, 72.78155996741175, 1422203235662.1047, 7233966937.994184, 1265980626171.4304, 5717.759288025154, 60.03447777895762, 387570278.4985512, 1213837.4110947177], 'rmse': [1.177540033408325, 0.953305074960843, 4.531312443155951, 9.594474861034442, 8.531210932066546, 1192561.6276159922, 85052.73033826829, 1125158.044974763, 75.61586664202926, 7.748191903854577, 19686.804679748086, 1101.7428970021624]}, 1385.4214319745597: {'perm': (2, 7, 4, 0, 6, 1, 8, 5, 3, 9, 10, 11), 'train_time': 485.9955813884735, 'predict_time': 0.28536319732666016, 'mean_abs_err': [2.7865885419911347, 969524.5419655278, 27.350136433732523, 28.668153859458815, 84594.35376517131, 2.791137651673889, 84545.26926495423, 969544.5360175238, 47.07861789121798, 4.812456695535601, 12197.756551516153, 570.3133328382962], 'mean_squ_err': [10.36246487954847, 1422444163107.2485, 1422.1240673314283, 1781.22269029514, 7240898663.690995, 12.04491927085999, 7233973971.79609, 1422722164517.6147, 2722.685052019315, 86.99341365573791, 376317360.1468299, 723758.9497507092], 'rmse': [3.2190782655208108, 1192662.6359148042, 37.7110602785367, 42.204534001634705, 85093.47015894343, 3.470579097335197, 85052.77168791203, 1192779.1767622433, 52.179354653151044, 9.327025981294247, 19398.90100358342, 850.7402363534413]}}
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

        perm_gp = PermutationGPModel(kernel='perm_eq',
                                     distance_kind=distance_kind,
                                     input_dim=num_permuted_dimensions)

        _log.info("Fitting hyperparameters of the permutation surrogate model")
        perm_gp = perm_gp.condition_on(train_permutations,
                                       train_log_probs)

        try:
            perm_gp.fit(optimizer_restarts=optimizer_restarts,
                        optimizer='l-bfgs-b',
                        trace=True,
                        err_level='raise',
                        median_heuristic_only=median_heuristic_only)

        except Exception as e:
            _log.exception("Training the surrogate model failed: {}".format(str(e)))
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

        flatten = lambda l: [item for sublist in l for item in sublist]

        def generate_candidates(num_candidates, dist=2):
            candidates = []

            s = 0
            while s < num_candidates:
                samp = uniform_gm.sample(as_tuple=True)

                if samp in observed_perms:
                    continue

                candidates.append(samp)
                s = s + 1

            # Generate the whole neighbourhood around the candidates
            candidates = flatten(list(map(lambda p: generate_neighbourhood(p, dist=dist), candidates)))

            # Eliminate duplicates
            return list(set(candidates))

        for sample_number in range(num_bayesopt_steps):

            # Generate 10000 candidate permutations
            candidate_perms = generate_candidates(num_bayesopt_candidates)

            # Calculate statistics for candidate permutations
            _log.info("Calculating statistics for candidate permutations")
            perm_means, perm_vars = perm_gp.predict(candidate_perms)
            perm_stds = tf.math.sqrt(perm_vars)

            _log.info("Calculcating EI acquisition function")
            max_observed_log_prob = tf.reduce_max(observed_log_probs)
            max_observed_log_prob = tf.cast(max_observed_log_prob, tf.float64)

            z = perm_means - max_observed_log_prob - xi

            z_ = tf.where(perm_vars > 1e-10, z / perm_stds, 0.)

            expected_imp = z * standard_normal.cdf(z_) + perm_stds * standard_normal.prob(z_)
            expected_imp = tf.where(perm_vars > 1e-10, expected_imp, 0.)

            next_evaluation_point = candidate_perms[tf.argmax(expected_imp).numpy()[0]]
            next_evaluation_point = tuple([int(x) for x in next_evaluation_point])

            # extend the permutation to include the target dimensions
            extended_next_evaluation_point = next_evaluation_point + target_dims

            bayesopt_stat = {"perm": extended_next_evaluation_point}

            _log.info(f"BayesOpt training round: {index + 1}/{rounds} for training set size {train_size}, "
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
                model.fit(optimizer_restarts=optimizer_restarts,
                                               optimizer=optimizer,
                                               trace=True)
            except Exception as e:
                _log.exception("Training failed: {}".format(str(e)))
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
                                                     target_dims=target_dims,
                                                     use_conditioning_data=True,
                                                     numpy=True)

            except Exception as e:
                raise e

            bayesopt_stat['predict_time'] = time.time() - start_time

            diff = (validate[outputs].values - mean)

            bayesopt_stat['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
            bayesopt_stat['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
            bayesopt_stat['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()
            bayesopt_stat['validation_log_prob'] = validation_log_prob

            bayesopt_stats[str(extended_next_evaluation_point)] = bayesopt_stat

            # Append the observed stuff to our list
            observed_perms.append(next_evaluation_point)
            observed_log_probs.append(validation_log_prob)

            # -----------------------------------------------------------------------------
            # Re-train model hyperparameters on newly observed data
            # -----------------------------------------------------------------------------
            _log.info("Retraining GP with newly observed data!")
            perm_gp = perm_gp.condition_on(observed_perms, observed_log_probs, keep_previous=False)
            perm_gp.fit(optimizer_restarts=optimizer_restarts,
                                             optimizer='l-bfgs-b',
                                             trace=True,
                                             median_heuristic_only=median_heuristic_only)

        experiment["bayesopt_stats"] = bayesopt_stats

        # The best permutation is the one with the highest log probability
        best_perm = observed_perms[tf.argmax(observed_log_probs).numpy()]
        best_perm = best_perm + target_dims

        best_log_prob = tf.reduce_max(observed_log_probs)

        experiment["perm"] = str([int(b) for b in best_perm])

        # ---------------------------------------------------------------------
        # Step 4: Retrain the model on the joint training-validation set
        # ---------------------------------------------------------------------
        _log.info(f"Retraining model with best permutation: {best_perm}, log_prob: {best_log_prob}")
        start_time = time.time()
        try:
            model = model.condition_on(train_and_validate[inputs].values,
                                       train_and_validate[outputs].values[:, best_perm],
                                       keep_previous=False)
            model.fit(optimizer_restarts=optimizer_restarts,
                                           optimizer=optimizer,
                                           trace=True)
        except Exception as e:
            _log.exception("Training failed: {}".format(str(e)))
            raise e

        experiment['train_time'] = time.time() - start_time

        # -----------------------------------------------------------------------------
        # Test best model
        # -----------------------------------------------------------------------------
        start_time = time.time()
        try:
            mean, _ = model.predict(test[inputs].values, numpy=True)
        except Exception as e:
            raise e

        experiment['predict_time'] = time.time() - start_time

        diff = (test[outputs].values - mean)

        experiment['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
        experiment['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
        experiment['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()

        experiments.append(experiment)

        _log.info("Saving experiments to {}".format(log_path))
        with open(log_path, mode='w') as out_file:
            json.dump(experiments, out_file, sort_keys=True, indent=4)

    return experiments


@ex.automain
def main(model, kernel, initialization, verbose, search_mode, latent_dim=None):
    data = load_dataset()

    df, input_labels, output_labels = prepare_gpar_data(data)

    if model == 'gpar':
        surrogate_model = GPARModel(kernel=kernel,
                                    input_dim=len(input_labels),
                                    output_dim=len(output_labels),
                                    initialization_heuristic=initialization,
                                    verbose=verbose)

    elif model == 'mf-gpar':
        surrogate_model = MatrixFactorizedGPARModel(kernel=kernel,
                                                    input_dim=len(input_labels),
                                                    output_dim=len(output_labels),
                                                    latent_dim=latent_dim,
                                                    initialization_heuristic=initialization,
                                                    verbose=verbose)

    elif model == 'p-gpar':
        surrogate_mmodel = PermutedGPARModel(kernel=kernel,
                                             input_dim=len(input_labels),
                                             output_dim=len(output_labels),
                                             initialization_heuristic=initialization,
                                             verbose=verbose)

    if search_mode == "random_search":
        results = run_random_experiment(model=model,
                                        data=df,
                                        inputs=input_labels,
                                        outputs=output_labels)

    elif search_mode == "greedy_search":
        run_greedy_experiment(model=model,
                              data=df,
                              inputs=input_labels,
                              outputs=output_labels)

    elif search_mode == "hbo":
        run_hierarchical_bayesopt_experiment(model=model,
                                             data=df,
                                             inputs=input_labels,
                                             outputs=output_labels)

    else:
        raise NotImplementedError
