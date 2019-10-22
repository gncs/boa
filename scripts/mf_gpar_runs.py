import argparse
import json
import time
from typing import Sequence

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from boa.models.mf_gpar import MFGPARModel
from .dataset_loader import load_dataset


def get_num_trainable_vars():
    return int(np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))


def pca_gpar_run(data, inputs: Sequence[str], outputs: Sequence[str], seed: int):
    runs = []

    for latent_size in [2, 5, 10, 15]:
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)

        model = MFGPARModel(kernel='matern', num_optimizer_restarts=5, verbose=True, latent_size=latent_size)

        for size in [25, 50, 100, 150, 200]:
            for repeat in range(5):
                run = {
                    'index': repeat,
                    'size': size,
                    'inputs': inputs,
                    'outputs': outputs,
                    'latent_size': latent_size,
                }

                train, test = train_test_split(data, train_size=size, test_size=200, random_state=seed + repeat)

                model.set_data(xs=train[inputs].values, ys=train[outputs].values)

                start_time = time.time()
                try:
                    model.train()
                except Exception:
                    print('Skipping...')
                    continue
                run['train_time'] = time.time() - start_time

                start_time = time.time()
                mean, _ = model.predict_batch(xs=test[inputs].values)
                run['predict_time'] = time.time() - start_time

                diff = (test[outputs].values - mean)
                run['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
                run['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
                run['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()
                run['num_trainable_vars'] = get_num_trainable_vars()

                runs.append(run)

    return runs


def main(args):
    data = load_dataset(path=args.dataset, kind=args.type)

    targets = ['avg_power', 'cycle', 'total_area']
    output_labels = data.output_labels.copy()

    # Append targets
    for target in targets:
        output_labels.remove(target)
        output_labels.append(target)

    results = pca_gpar_run(data.df, inputs=data.input_labels, outputs=output_labels, seed=42)

    with open('mf_gpar_runs.json', mode='w') as f:
        json.dump(results, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='path to file containing dataset', required=True)
    parser.add_argument('--type', help='select dataset type', required=True)

    main(parser.parse_args())
