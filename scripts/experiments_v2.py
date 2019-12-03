import logging
import argparse
import json
import time
import os
from typing import Sequence

import numpy as np

from sklearn.model_selection import train_test_split

from boa.models.abstract_model_v2 import AbstractModel
from boa.models.fully_factorized_gp_v2 import FullyFactorizedGPModel
from boa.models.gpar_v2 import GPARModel
from boa.models.matrix_factorized_gpar_v2 import MatrixFactorizedGPARModel

from boa.core import GaussianProcess, setup_logger

from dataset_loader import load_dataset

import tensorflow as tf

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/experiments.log")

AVAILABLE_DATASETS = ["fft", "stencil3d"]

LOG_LEVELS = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "warn": logging.WARNING
}


def run_experiment(model,
                   data,
                   optimizer_restarts,
                   inputs: Sequence[str],
                   outputs: Sequence[str],
                   logdir,
                   experiment_file_name,
                   matrix_factorized,
                   rounds: int = 5,
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

    for size in [25, 50, 100, 150, 200]:
        for index in range(rounds):

            if verbose:
                logger.info("-----------------------------------------------------------")
                logger.info(f"Training round: {index + 1} for training set size {size}")
                logger.info("-----------------------------------------------------------")

            experiment = {'index': index,
                          'size': size,
                          'inputs': inputs,
                          'outputs': outputs}

            if matrix_factorized:
                experiment["latent_size"] = model.latent_dim

            train, test = train_test_split(data,
                                           train_size=size,
                                           test_size=200,
                                           random_state=seed + index)

            start_time = time.time()
            try:
                model = model.condition_on(train[inputs].values, train[outputs].values[:, :], keep_previous=False)
                model.fit_to_conditioning_data(optimizer_restarts=optimizer_restarts)
            except Exception as e:
                logger.exception("Training failed: {}".format(str(e)))
                raise e

            experiment['train_time'] = time.time() - start_time

            start_time = time.time()

            try:
                mean, _ = model.predict(test[inputs].values, numpy=True)
            except Exception as e:
                logger.exception("Prediction failed: {}, saving model!".format(str(e)))

                model.save("models/exceptions/" + model.name + "/model")
                raise e
                # continue

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


def prepare_ff_gp_data(data):
    return data.df, data.input_labels.copy(), data.output_labels.copy()


def prepare_ff_gp_aux_data(data,
                           targets):
    inputs = data.input_labels + data.output_labels

    for x in targets:
        inputs.remove(x)

    return data.df, inputs, targets


def prepare_gpar_data(data,
                      targets):
    output_labels = data.output_labels.copy()

    for target in targets:
        output_labels.remove(target)
        output_labels.append(target)

    return data.df, data.input_labels.copy(), output_labels


def main(args,
         seed=27,
         experiment_json_format="{}_experiments.json",
         targets=('avg_power', 'cycle', 'total_area')):
    data = load_dataset(path=args.dataset, kind=args.task)

    model = None

    if args.model == 'ff-gp':
        df, input_labels, output_labels = prepare_ff_gp_data(data)
        df_aux, input_labels_aux, output_labels_aux = prepare_ff_gp_aux_data(data,
                                                                             targets)

        model = FullyFactorizedGPModel(kernel=args.kernel,
                                       input_dim=len(input_labels),
                                       output_dim=len(output_labels),
                                       initialization_heuristic="median",
                                       verbose=args.verbose)

        # Perform experiments
        results = run_experiment(model=model,
                                 data=df,
                                 optimizer_restarts=args.num_optimizer_restarts,
                                 inputs=input_labels,
                                 outputs=output_labels,
                                 matrix_factorized=False,
                                 logdir=args.logdir,
                                 experiment_file_name=experiment_json_format.format(args.model),
                                 seed=seed,
                                 rounds=5,
                                 verbose=args.verbose)

        # For the fully factorized GP model,
        # we also look at training using auxiliary data
        # logger.info("============================================================")
        # logger.info("Performing auxiliary training for FF-GP")
        # logger.info("============================================================")
        #
        # model = FullyFactorizedGPModel(kernel=args.kernel,
        #                                input_dim=len(input_labels_aux),
        #                                output_dim=len(output_labels_aux),
        #                                initialization_heuristic="median",
        #                                verbose=args.verbose)
        #
        # print(input_labels_aux)
        #
        # results = run_experiment(model=model,
        #                          data=df_aux,
        #                          optimizer_restarts=args.num_optimizer_restarts,
        #                          inputs=input_labels_aux,
        #                          outputs=output_labels_aux,
        #                          matrix_factorized=False,
        #                          logdir=args.logdir,
        #                          experiment_file_name=experiment_json_format.format(args.model),
        #                          seed=seed,
        #                          rounds=5,
        #                          verbose=args.verbose)

    elif args.model in ["gpar", "mf-gpar"]:
        df, input_labels, output_labels = prepare_gpar_data(data,
                                                            targets)

        if args.model == 'gpar':
            model = GPARModel(kernel=args.kernel,
                              input_dim=len(input_labels),
                              output_dim=len(output_labels),
                              initialization_heuristic="median",
                              verbose=args.verbose)

        elif args.model == 'mf-gpar':
            model = MatrixFactorizedGPARModel(kernel=args.kernel,
                                              input_dim=len(input_labels),
                                              output_dim=len(output_labels),
                                              latent_dim=args.latent_dim,
                                              initialization_heuristic="median",
                                              verbose=args.verbose)

        # Perform experiments
        results = run_experiment(model=model,
                                 data=df,
                                 optimizer_restarts=args.num_optimizer_restarts,
                                 inputs=input_labels,
                                 outputs=output_labels,
                                 logdir=args.logdir,
                                 matrix_factorized=args.model == "mf-gpar",
                                 experiment_file_name=experiment_json_format.format(args.model),
                                 seed=seed,
                                 rounds=5,
                                 verbose=args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-D', type=str, required=True,
                        help="Path to the dataset.")

    parser.add_argument('--task', '-T', choices=AVAILABLE_DATASETS, required=True,
                        help="Task for which we are providing the dataset.")

    parser.add_argument('--logdir', type=str, default="logs",
                        help="Path to the directory to which we will write the log files "
                             "for the experiment.")

    parser.add_argument('--loglevel', choices=LOG_LEVELS, )

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

    # =========================================================================
    # Matrix factorized GPAR
    # =========================================================================

    mf_gpar_mode = model_subparsers.add_parser("mf-gpar",
                                               formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                               description="use a GPAR Model with factorized length scale matrix")

    mf_gpar_mode.add_argument("--latent_dim", type=int, default=5,
                              help="Effective dimension of the factorization.")

    # Add common options to models
    for mode in [ff_gp_mode, gpar_mode, mf_gpar_mode]:
        mode.add_argument("--kernel",
                          choices=GaussianProcess.AVAILABLE_KERNELS,
                          default="matern52",
                          help="GP kernel to use.")

        mode.add_argument("--num_optimizer_restarts",
                          type=int,
                          default=5,
                          help="Number of random initializations to try in a single training cycle.")

    args = parser.parse_args()

    main(args)
