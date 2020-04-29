"""
Verify that different permutations can have a significant effect on GPAR's performance
on the target dimensions.
"""
import json
import os
import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from boa import ROOT_DIR
from boa.core import transform_df
from boa.models import GPARModel

from dataset_config import dataset_ingredient, load_dataset

# Set CPU as available physical device
tf.config.experimental.set_visible_devices([], 'GPU')

tfd = tfp.distributions

ex = Experiment('permutation_effects', ingredients=[dataset_ingredient])
database_url = "127.0.0.1:27017"
database_name = "boa_fitting_experiments"
ex.captured_out_filter = apply_backspaces_and_linefeeds

ex.observers.append(MongoObserver(url=database_url, db_name=database_name))


@ex.config
def experiment_config(dataset):
    task = dataset["name"]

    use_input_transforms = True

    num_targets = len(dataset["targets"])

    # Number of experiments to perform for the same permutation
    rounds = 25

    # Number of permutations to draw
    num_permutations = 25

    train_dataset_size = 50
    test_dataset_size = 200

    denoising = False
    fit_joint = False
    iters = 1000

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Path to the directory to which we will write the log files
    log_path = f"{ROOT_DIR}/../logs/{task}/ordering/effect/{current_time}/" \
               f"permutations_{num_permutations}_rounds_{rounds}_data_{train_dataset_size}.json"

    # GP kernel to use.
    kernel = "matern52"

    # Number of random initializations to try in a single training cycle.
    num_optimizer_restarts = 3

    # Optimization algorithm to use when fitting the models' hyperparameters.
    optimizer = "l-bfgs-b"

    # Initialization heuristic for the hyperparameters of the models.
    initialization = "l2_median"

    verbose = True


@ex.automain
def experiment(
    dataset,
    rounds,
    num_permutations,
    num_targets,
    train_dataset_size,
    test_dataset_size,
    use_input_transforms,
    kernel,
    initialization,
    optimizer,
    num_optimizer_restarts,
    iters,
    fit_joint,
    denoising,
    log_path,
    verbose,
    _seed,
    _log,
):
    log_dir = os.path.dirname(log_path)

    # Make sure the directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    experiments = []

    input_labels = np.array(dataset["input_labels"])
    output_labels = np.array(dataset["output_labels"])

    input_dim = len(input_labels)
    output_dim = len(output_labels)

    def extend_perm(perm):

        if isinstance(perm, tf.Tensor):
            perm = perm.numpy()

        if isinstance(perm, np.ndarray):
            perm = perm.tolist()

        return tuple(perm + list(range(len(perm), output_dim)))

    # Load in the appropriate dataset
    ds = load_dataset()
    data = ds.df

    # Transform the data before it is passed to anything.
    if use_input_transforms:
        data = transform_df(data, dataset["input_transforms"])

    model = GPARModel(kernel=kernel, input_dim=input_dim, output_dim=output_dim, verbose=verbose)

    # Set seed for reproducibility
    np.random.seed(_seed)
    tf.random.set_seed(_seed)

    # Draw the permutations uniformly (for the non-target dimensions only!)
    uniform_perm_dist = tfd.PlackettLuce(tf.ones(output_dim - num_targets))
    permutations = uniform_perm_dist.sample(num_permutations).numpy()

    # Will contain the experiment statistics and will be saved to a JSON file
    experiments = []

    for index in range(rounds):
        _log.info(f"Performing round {index + 1}/{rounds}!")

        # Split the data differently for each round
        train_data, test_data = train_test_split(data,
                                                 train_size=train_dataset_size,
                                                 test_size=test_dataset_size,
                                                 random_state=_seed + index)

        for perm_index, permutation in enumerate(permutations):

            # Extend perm returns a tuple, which is serializable
            permutation = extend_perm(permutation)
            permuted_output_labels = output_labels[np.array(permutation)]

            if verbose:
                _log.info("-----------------------------------------------------------")
                _log.info(f"Training permutation: {permutation}. ({perm_index + 1}/{len(permutations)})")
                _log.info("-----------------------------------------------------------")

            experiment = {
                'index': index,
                'inputs': input_labels.tolist(),
                'outputs': permuted_output_labels.tolist(),
                'permutation': permutation,
            }

            perm_model = model.condition_on(train_data[input_labels].values, train_data[permuted_output_labels].values)

            perm_model.fit(fit_joint=fit_joint,
                           map_estimate=False,
                           length_scale_init_mode=initialization,
                           optimizer=optimizer,
                           optimizer_restarts=num_optimizer_restarts,
                           iters=iters,
                           denoising=denoising,
                           trace=True,
                           err_level="raise")

            mean, variance = perm_model.predict(test_data[input_labels].values,
                                                marginalize_hyperparameters=False,
                                                numpy=True)

            diff = (test_data[permuted_output_labels].values - mean)

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
