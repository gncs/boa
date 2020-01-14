import argparse
import json
import time
from typing import Sequence

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from boa.models.gpar import GPARModel
from boa.datasets.loader import load_dataset


def gp_run(data, inputs: Sequence[str], outputs: Sequence[str], seed: int):
    runs = []

    np.random.seed(seed)
    tf.set_random_seed(seed)
    tf.reset_default_graph()

    model = GPARModel(kernel='matern', num_optimizer_restarts=5, verbose=True)

    for size in [25, 50, 100, 150, 200]:
        for prop in outputs:
            print(f'Property: {prop}')
            for repeat in range(5):
                run = {'index': repeat, 'size': size, 'inputs': inputs, 'output': prop}

                train, test = train_test_split(data, train_size=size, test_size=200, random_state=seed + repeat)

                model.set_data(xs=train[inputs].values, ys=train[[prop]].values)

                start_time = time.time()
                try:
                    model.train()
                except Exception:
                    print('Failed to train model... Skipping.')
                    continue
                run['train_time'] = time.time() - start_time

                with tf.Session() as session:
                    model.load_model(session)
                    run['kern_variance'] = np.exp(session.run(model.log_hps[0][0])).tolist()
                    run['noise_variance'] = np.exp(session.run(model.log_hps[0][1])).tolist()
                    run['lengthscales'] = np.exp(session.run(model.log_hps[0][2])).tolist()

                start_time = time.time()
                mean, _ = model.predict_batch(xs=test[inputs].values)
                run['predict_time'] = time.time() - start_time

                diff = (test[[prop]].values - mean)[:, 0]
                run['mean_abs_err'] = np.mean(np.abs(diff))
                run['mean_squ_err'] = np.mean(np.square(diff))
                run['rmse'] = np.sqrt(np.mean(np.square(diff)))

                runs.append(run)

    return runs


def main(args):
    data = load_dataset(path=args.dataset, kind=args.type)
    seed = 42

    # Normal GP runs
    # Run for all output labels for PCA analysis later
    results = gp_run(data.df, inputs=data.input_labels, outputs=data.output_labels, seed=seed)

    with open('gp_runs.json', mode='w') as f:
        json.dump(results, f, sort_keys=True, indent=4)

    # GP runs with auxiliary data
    inputs = data.input_labels + data.output_labels

    targets = ['avg_power', 'cycle', 'total_area']
    for x in targets:
        inputs.remove(x)

    results = gp_run(data.df, inputs=inputs, outputs=targets, seed=seed)

    with open('gp_aux_runs.json', mode='w') as f:
        json.dump(results, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='path to file containing dataset', required=True)
    parser.add_argument('--type', help='select dataset type', required=True)

    main(parser.parse_args())
