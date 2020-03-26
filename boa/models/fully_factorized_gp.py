import logging

import json

from boa.core.utils import setup_logger
from .multi_output_gp_regression_model import MultiOutputGPRegressionModel, ModelError
from boa import ROOT_DIR


__all__ = ["FullyFactorizedGPModel"]

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file=f"{ROOT_DIR}/../logs/ff_gp.log")


class FullyFactorizedGPModel(MultiOutputGPRegressionModel):

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
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

    def gp_input(self, index, xs, ys):
        return xs

    def gp_input_dim(self, index):
        return self.input_dim

    def gp_output(self, index, ys):
        return ys[:, index:index + 1]

    def has_explicit_length_scales(self):
        return True

    def gp_predictive_input(self, xs, means):
        return xs

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
            "parallel": self.parallel,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_config(config, **kwargs):
        return FullyFactorizedGPModel(**config)
