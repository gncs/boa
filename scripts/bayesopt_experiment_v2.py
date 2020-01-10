import logging
import argparse
import os

from typing import List

import pandas as pd
import tensorflow as tf

from boa.core.utils import setup_logger
from boa.objective.abstract import AbstractObjective

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="bayesopt_experiment_v2")

objective_labels = ['cycle', 'avg_power', 'total_area']

AVAILABLE_DATASETS = ["fft",
                      "stencil3d"]


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


def main(args):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        "-D",
                        help="Path to file containing dataset",
                        required=True)

    parser.add_argument("--type", "-T", help="Dataset type", choices=AVAILABLE_DATASETS)
