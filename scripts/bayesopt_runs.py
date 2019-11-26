# Bayesian Optimization of FFT Benchmark
import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from boa.acquisition.loader import load_acquisition
from boa.models.loader import load_model
from boa.objective.abstract import AbstractObjective
from boa.optimization.data import generate_data, Data, FileHandler
from boa.optimization.loader import load_optimizer
from scripts.dataset_loader import DataTuple, load_dataset

objective_labels = ['cycle', 'avg_power', 'total_area']


class Objective(AbstractObjective):
    def __init__(self, df: pd.DataFrame, input_labels: List[str], output_labels: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = df
        self.input_labels = input_labels
        self.output_labels = output_labels

    def get_input_labels(self) -> List[str]:
        """Return input labels as a list of length D_input"""
        return self.input_labels

    def get_output_labels(self) -> List[str]:
        """Return output labels as a list of length D_output"""
        return self.output_labels

    def get_candidates(self) -> np.ndarray:
        """Return potential candidates as an array of shape N x D_input"""
        return self.data[self.input_labels].values

    def __call__(self, value: np.ndarray) -> np.ndarray:
        """Return output of objective function as an array of shape N x D_output"""
        mask = pd.Series([True] * self.data.shape[0])
        for k, v in zip(self.input_labels, value):
            mask = mask & (self.data[k].values == v)

        assert (mask.sum() == 1)

        return self.data.loc[mask, self.output_labels].values


def optimize(objective, model_config: dict, opt_config: dict, acq_config: dict, seed: int) -> Data:
    init_data_size = 10

    model = load_model(model_config)
    acq = load_acquisition(acq_config)
    optimizer = load_optimizer(opt_config)
    candidates = objective.get_candidates()

    data = generate_data(objective=objective, size=init_data_size, seed=seed)
    model._set_data(xs=data.input, ys=data.output)
    model.fit()

    xs, ys = optimizer.optimize(
        f=objective,
        model=model,
        acq_fun=acq,
        xs=data.input,
        ys=data.output,
        candidate_xs=candidates,
    )

    return Data(xs=xs, ys=ys, x_labels=objective.input_labels, y_labels=objective.output_labels)


def get_default_opt_config() -> dict:
    return {
        'name': 'default',
        'parameters': {
            'max_num_iterations': 120,
            'batch_size': 1,
            'verbose': True,
        }
    }


def get_default_acq_config(df: pd.DataFrame) -> dict:
    max_values = df[objective_labels].apply(max).values

    return {
        'name': 'sms-ego',
        'parameters': {
            'gain': 1,
            'epsilon': 0.01,
            'reference': max_values,
            'output_slice': (-3, None),
        }
    }


def random_opt(dt: DataTuple, seed: int) -> Data:
    model_config = {
        'name': 'random',
        'parameters': {
            'seed': seed,
            'num_samples': 10,
        }
    }

    return optimize(objective=Objective(df=dt.df, input_labels=dt.input_labels, output_labels=objective_labels),
                    model_config=model_config,
                    opt_config=get_default_opt_config(),
                    acq_config=get_default_acq_config(df=dt.df),
                    seed=seed)


def gp_opt(dt: DataTuple, seed: int) -> Data:
    model_config = {
        'name': 'gp',
        'parameters': {
            'kernel': 'matern',
            'num_optimizer_restarts': 3,
            'parallel': True,
        }
    }

    tf.reset_default_graph()
    tf.set_random_seed(seed)

    return optimize(objective=Objective(df=dt.df, input_labels=dt.input_labels, output_labels=objective_labels),
                    model_config=model_config,
                    opt_config=get_default_opt_config(),
                    acq_config=get_default_acq_config(df=dt.df),
                    seed=seed)


def gpar_opt(dt: DataTuple, seed: int) -> Data:
    model_config = {
        'name': 'gpar',
        'parameters': {
            'kernel': 'matern',
            'num_optimizer_restarts': 3,
            'verbose': True,
        }
    }

    tf.reset_default_graph()
    tf.set_random_seed(seed)

    # Append targets
    labels = dt.output_labels.copy()
    for objective in objective_labels:
        labels.remove(objective)
        labels.append(objective)

    output_labels = labels

    return optimize(objective=Objective(df=dt.df, input_labels=dt.input_labels, output_labels=output_labels),
                    model_config=model_config,
                    opt_config=get_default_opt_config(),
                    acq_config=get_default_acq_config(df=dt.df),
                    seed=seed)


def mf_gpar_opt(dt: DataTuple, seed: int) -> Data:
    model_config = {
        'name': 'mf-gpar',
        'parameters': {
            'kernel': 'matern',
            'num_optimizer_restarts': 3,
            'verbose': True,
            'latent_size': 5,
        }
    }

    tf.reset_default_graph()
    tf.set_random_seed(seed)

    # Append targets
    labels = dt.output_labels.copy()
    for objective in objective_labels:
        labels.remove(objective)
        labels.append(objective)

    output_labels = labels

    return optimize(objective=Objective(df=dt.df, input_labels=dt.input_labels, output_labels=output_labels),
                    model_config=model_config,
                    opt_config=get_default_opt_config(),
                    acq_config=get_default_acq_config(df=dt.df),
                    seed=seed)


def main(args):
    fun_to_name = {
        random_opt: 'random',
        gp_opt: 'gp',
        gpar_opt: 'gpar',
        mf_gpar_opt: 'mf-gpar',
    }

    data_tuple = load_dataset(path=args.dataset, kind=args.type)

    for f, name in fun_to_name.items():
        for i in range(5):
            run = f(dt=data_tuple, seed=i)
            os.makedirs(name, exist_ok=True)
            handler = FileHandler(path=os.path.join(name, f'{i}.json'))
            handler.save(run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='path to file containing dataset', required=True)
    parser.add_argument('--type', help='select dataset type', required=True)

    main(parser.parse_args())
