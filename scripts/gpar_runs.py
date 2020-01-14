import argparse
import json
import time
from typing import Sequence

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from boa.models.gpar import GPARModel
from boa.datasets.loader import load_dataset


def gpar_run(data, inputs: Sequence[str], outputs: Sequence[str], seed: int):
    runs = []

    np.random.seed(seed)
    tf.set_random_seed(seed)
    model = GPARModel(kernel='matern', num_optimizer_restarts=5, verbose=True)

    for size in [25, 50, 100, 150, 200]:
        for repeat in range(5):
            run = {'index': repeat, 'size': size, 'inputs': inputs, 'outputs': outputs}

            train, test = train_test_split(data, train_size=size, test_size=200, random_state=seed + repeat)

            model.set_data(xs=train[inputs].values, ys=train[outputs].values)

            start_time = time.time()
            try:
                model.train()
            except Exception:
                print('Failed to train model... Skipping.')
                continue
            run['train_time'] = time.time() - start_time

            start_time = time.time()
            mean, _ = model.predict_batch(xs=test[inputs].values)
            run['predict_time'] = time.time() - start_time

            diff = (test[outputs].values - mean)
            run['mean_abs_err'] = np.mean(np.abs(diff), axis=0).tolist()
            run['mean_squ_err'] = np.mean(np.square(diff), axis=0).tolist()
            run['rmse'] = np.sqrt(np.mean(np.square(diff), axis=0)).tolist()

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

    results = gpar_run(data.df, inputs=data.input_labels, outputs=output_labels, seed=42)

    with open('gpar_runs.json', mode='w') as f:
        json.dump(results, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='path to file containing dataset', required=True)
    parser.add_argument('--type', help='select dataset type', required=True)

    main(parser.parse_args())
