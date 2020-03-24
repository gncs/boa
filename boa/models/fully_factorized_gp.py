import logging

import json

from typing import List

import tensorflow as tf
import numpy as np

from boa.core.gp import GaussianProcess
from boa.core.utils import setup_logger
from .abstract_model import AbstractModel, ModelError
from boa import ROOT_DIR

from not_tf_opt import minimize, BoundedVariable

__all__ = ["FullyFactorizedGPModel"]

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file=f"{ROOT_DIR}/../logs/ff_gp.log")


class FullyFactorizedGPModel(AbstractModel):

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 initialization_heuristic: str = "median",
                 parallel: bool = False,
                 name: str = "gp_model",
                 verbose=True,
                 **kwargs):

        super(FullyFactorizedGPModel, self).__init__(kernel=kernel,
                                                     input_dim=input_dim,
                                                     output_dim=output_dim,
                                                     parallel=parallel,
                                                     verbose=verbose,
                                                     name=name,
                                                     **kwargs)

        self.initialization_heuristic = initialization_heuristic

        # Create GP hyperparameter variables
        for i in range(self.output_dim):
            self.length_scales.append(
                tf.Variable(tf.ones(self.input_dim, dtype=tf.float64), name=f"{i}/length_scales", trainable=False))

            self.signal_amplitudes.append(
                tf.Variable((1.0,), dtype=tf.float64, name=f"{i}/signal_amplitude", trainable=False))

            self.noise_amplitudes.append(
                tf.Variable((1.0,), dtype=tf.float64, name=f"{i}/noise_amplitude", trainable=False))

    def gp_input(self, index, xs, ys):
        return xs

    def gp_input_dim(self, index):
        return self.input_dim

    def gp_output(self, index, ys):
        return ys[:, index:index + 1]

    @staticmethod
    def restore(save_path):

        with open(save_path + ".json", "r") as config_file:
            config = json.load(config_file)

        model = FullyFactorizedGPModel.from_config(config, )

        model.load_weights(save_path)
        model.create_gps()

        return model

    def get_config(self):

        return {
            "name": self.name,
            "kernel": self.kernel_name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "initialization_heuristic": self.initialization_heuristic,
            "parallel": self.parallel,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_config(config, **kwargs):
        return FullyFactorizedGPModel(**config)
