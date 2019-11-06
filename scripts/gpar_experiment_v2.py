import argparse
import json
import time
import os
from typing import Sequence

import numpy as np
import tensorflow as tf

from boa.models.gpar_v2 import GPARModel
from dataset_loader import load_dataset

AVAILABLE_DATASETS = ["fft", "stencil3d"]


def gp_experiment(data,
                  inputs: Sequence[str],
                  outputs: Sequence[str],
                  rounds: int = 5,
                  seed: int = 42):

    experiments = []

    # Set seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Prepare the dataset
    #ds = tf.data.Dataset.from_tensor_slices(data)

    model = GPARModel(kernel='matern52', num_optimizer_restarts=5, verbose=True)

    for size in [25, 50, 100, 150, 200]:

        for output in outputs:
            print(f'Property: {output}')

            for index in range(rounds):
                experiment = {'index': index,
                              'size': size,
                              'inputs': inputs,
                              'output': output}

    return experiments


def main(args, seed=42, gp_experiment_json_name="gp_experiments.json"):
    data = load_dataset(path=args.dataset, kind=args.task)

    print(data)

    results = gp_experiment(data=data.df,
                            inputs=data.input_labels,
                            outputs=data.output_labels,
                            seed=seed)

    gp_experiment_file_path = os.path.join(args.logdir, gp_experiment_json_name)

    # Make sure the directory exists
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    with open(gp_experiment_file_path, mode='w') as out_file:
        json.dump(results, out_file, sort_keys=True, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-D', type=str, required=True,
                        help="Path to the dataset.")

    parser.add_argument('--task', '-T', choices=AVAILABLE_DATASETS, required=True,
                        help="Task for which we are providing the dataset.")

    parser.add_argument('--logdir', type=str, default=".",
                        help="Path to the directory to which we will write the log files "
                             "for the experiment.")

    args = parser.parse_args()

    main(args)

