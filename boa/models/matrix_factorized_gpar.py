import logging
import json

import numpy as np
import tensorflow as tf

from tqdm import trange

from boa.models.abstract_model import ModelError
from .gpar import GPARModel

from boa.core.utils import setup_logger
from boa.core.gp import GaussianProcess

from not_tf_opt import BoundedVariable, minimize

from boa import ROOT_DIR

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file=f"{ROOT_DIR}/../logs/mf_gpar.log")


class MatrixFactorizedGPARModel(GPARModel):
    LLS_MAT = "left_length_scale_matrix"
    RLS_MAT = "right_length_scale_matrix"

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 latent_dim: int,
                 initialization_heuristic: str = "median",
                 verbose: bool = False,
                 name="matrix_factorized_gpar_model",
                 **kwargs):

        super(MatrixFactorizedGPARModel, self).__init__(
            kernel=kernel,
            input_dim=input_dim,
            output_dim=output_dim,
            initialization_heuristic=initialization_heuristic,
            _create_length_scales=False,  # Never create the length scales in the parent class
            verbose=verbose,
            name=name,
            **kwargs)

        self.latent_dim = latent_dim

        self.output_length_scales = []

        # Create TF variables for the hyperparameters

        # Create low rank representation for the input length scales
        self.left_length_scale_matrix = tf.Variable(tf.ones((self.output_dim, self.latent_dim), dtype=tf.float64),
                                                    trainable=False,
                                                    name=self.LLS_MAT)

        self.right_length_scale_matrix = tf.Variable(tf.ones((self.latent_dim, self.input_dim), dtype=tf.float64),
                                                     trainable=False,
                                                     name=self.RLS_MAT)

        # # dimensions O x I
        input_length_scales = tf.matmul(self.left_length_scale_matrix, self.right_length_scale_matrix)

        for i in range(self.output_dim):
            # Length scales for the output dimensions only
            out_length_scales = tf.Variable(tf.ones(i, dtype=tf.float64),
                                            name=f"{i}/output_length_scale",
                                            trainable=False)

            self.output_length_scales.append(out_length_scales)

            # # i-th length scales
            length_scales = tf.concat((input_length_scales[i, :], out_length_scales), axis=0)

            self.length_scales.append(length_scales)

    def copy(self, name=None):

        mf_gpar = super(MatrixFactorizedGPARModel, self).copy(name=name)

        input_length_scales = tf.matmul(mf_gpar.left_length_scale_matrix, mf_gpar.right_length_scale_matrix)

        for i in range(self.output_dim):
            mf_gpar.length_scales[i] = tf.concat((input_length_scales[i, :], mf_gpar.output_length_scales[i]), axis=0)

        return mf_gpar

    def initialize_all_hyperparameters(self,
                                       length_scale_init_mode: str,
                                       **kwargs):

        # Initialize the hyperparameters for regular GPAR
        all_hyperparams = super().initialize_all_hyperparameters(length_scale_init_mode=length_scale_init_mode,
                                                                 **kwargs)

        joint_length_scales, signal_amplitudes, noise_amplitudes = all_hyperparams

        # Separate the input and output length scales
        input_length_scales = []
        output_length_scales = []

        for i in range(self.output_dim):
            joint_ls = joint_length_scales[i]

            # Separate out output length scale
            output_length_scale = BoundedVariable(joint_ls()[self.input_dim:],
                                                  lower=joint_ls.lower,
                                                  upper=joint_ls.upper,
                                                  dtype=joint_ls.dtype)

            output_length_scales.append(output_length_scale)

            # Separate out input length scale
            input_length_scales.append(joint_ls()[:self.input_dim])

        # Create joint input length scale matrix
        ls_mat = tf.stack(input_length_scales)

        # SVD decomposition of the length scale matrix
        s, u, v = tf.linalg.svd(ls_mat)

        # Create low-rank approximation using Eckart-Young-Mirsky theorem
        s = tf.sqrt(s[:self.latent_dim], tf.float64)
        s_diag = tf.linalg.diag(s)

        # Left length scale initializer: U * sqrt(S)
        lls_init = tf.matmul(u[:, :self.latent_dim], s_diag)

        # Left length scale initializer: sqrt(S) * V^T
        rls_init = tf.matmul(s_diag, tf.transpose(v[:, :self.latent_dim]))

        # Create container for matrix factors:
        # Left length scale (LLS) matrix and Right length scale (RLS) matrix
        lls_mat = BoundedVariable(lls_init, lower=1e-10, upper=1e2, dtype=tf.float64)

        rls_mat = BoundedVariable(rls_init, lower=1e-10, upper=1e2, dtype=tf.float64)

        return lls_mat, rls_mat, output_length_scales, signal_amplitudes, noise_amplitudes

    def create_gps(self):
        self.models.clear()

        self.length_scales.clear()

        # dimensions O x I
        input_length_scales = tf.matmul(self.left_length_scale_matrix, self.right_length_scale_matrix)

        for i in range(self.output_dim):
            # i-th length scales
            length_scales = tf.concat((input_length_scales[i, :], self.output_length_scales[i]), axis=0)

            self.length_scales.append(length_scales)

        for i in range(self.output_dim):
            gp = GaussianProcess(kernel=self.kernel_name,
                                 input_dim=self.input_dim + i,
                                 signal_amplitude=self.signal_amplitudes[i],
                                 length_scales=self.length_scales[i],
                                 noise_amplitude=self.noise_amplitudes[i])

            self.models.append(gp)

    @staticmethod
    def restore(save_path):

        with open(save_path + ".json", "r") as config_file:
            config = json.load(config_file)

        model = GPARModel.from_config(config, )

        model.load_weights(save_path)
        model.create_gps()

        return model

    def get_config(self):

        return {
            "name": self.name,
            "kernel": self.kernel_name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "latent_dim": self.latent_dim,
            "denoising": self.denoising,
            "initialization_heuristic": self.initialization_heuristic,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_config(config, **kwargs):
        return MatrixFactorizedGPARModel(**config)
