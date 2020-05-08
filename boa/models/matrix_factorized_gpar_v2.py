import logging
import json

from typing import List

import numpy as np
import tensorflow as tf

from tqdm import trange

from boa.models.multi_output_gp_regression_model import ModelError
from .gpar import GPARModel

from boa.core.utils import setup_logger
from not_tf_opt import PositiveVariable, BoundedVariable, UnconstrainedVariable
from not_tf_opt import map_to_bounded_interval, map_from_bounded_interval

from boa import ROOT_DIR

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file=f"{ROOT_DIR}/../logs/mf_gpar.log")


class MatrixFactorizedGPARModelV2(GPARModel):
    LLS_MAT = "left_length_scale_matrix"
    RLS_MAT = "right_length_scale_matrix"

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 latent_dim: int,
                 verbose: bool = False,
                 name="matrix_factorized_gpar_model",
                 **kwargs):

        super(MatrixFactorizedGPARModelV2, self).__init__(
            kernel=kernel,
            input_dim=input_dim,
            output_dim=output_dim,
            verbose=verbose,
            name=name,
            **kwargs)

        self.latent_dim = latent_dim

        self._scale_coeff = PositiveVariable(0.1, dtype=tf.float64, name="scale_coefficient")

        # The product of these gives the log-lengthscales!
        self._left_ls_matrix = UnconstrainedVariable(tf.zeros([self.output_dim, self.latent_dim]),
                                                     name="left_length_scale_matrix")
        self._right_ls_matrix = UnconstrainedVariable(tf.zeros([self.latent_dim, self.input_dim]),
                                                      name="right_length_scale_matrix")

        self._input_ls_lower_bound = tf.Variable(tf.zeros([0, 0], dtype=tf.float64),
                                                 shape=(None, None),
                                                 dtype=tf.float64,
                                                 name="ls_lower_bound")

        self._input_ls_upper_bound = tf.Variable(tf.zeros([0, 0], dtype=tf.float64),
                                                 shape=(None, None),
                                                 dtype=tf.float64,
                                                 name="ls_upper_bound")

        self._output_length_scales: List[BoundedVariable] = [
            BoundedVariable(tf.zeros([self.gp_input_dim(i) - self.input_dim]),
                            lower=-np.inf,
                            upper=np.inf,
                            name=f"output_length_scales_{i}") for i in range(self.output_dim)
        ]

    @property
    def input_length_scales(self):
        ls_mat = tf.matmul(self._left_ls_matrix(), self._right_ls_matrix())

        # Use Frobenius norm because it is rotation invariant and easy to calculate
        squared_ls_matrix_norm = tf.reduce_sum(ls_mat * ls_mat)

        ls_mat = self._scale_coeff() * ls_mat / squared_ls_matrix_norm

        ls_mat = map_to_bounded_interval(ls_mat, lower=self._input_ls_lower_bound, upper=self._input_ls_upper_bound)

        return ls_mat

    def has_explicit_length_scales(self):
        return False

    def gp_length_scales(self, index):
        # Wrap in a lambda so it imitates the forward transform of a variable
        return lambda: tf.concat([self.input_length_scales[index, :], self._output_length_scales[index]()], axis=0)

    def length_scales(self):
        """
        Using this function is a much more efficient way of getting every length scale compared
        to calling self.gp_length_scales with every index.

        :return:
        """

        print(self._scale_coeff())

        input_length_scales = self.input_length_scales
        output_length_scales = self._output_length_scales

        length_scales = [tf.concat([input_length_scales[i, :], output_length_scales[i]()], axis=0)
                         for i in range(self.output_dim)]

        return list(map(lambda x: (lambda: x), length_scales))

    def gp_variables_to_train(self, index, transformed):
        raise ModelError("MF-GPAR model cannot be trained in a factorized manner!")

    def gp_assign_variables(self, index, values):
        raise ModelError("Variables for MF-GPAR model cannot be assigned " "in a factorized manner!")

    def variables_to_train(self, transformed):
        signal_amplitudes = [self.gp_signal_amplitude(i) for i in range(self.output_dim)]

        noise_amplitudes = [self.gp_noise_amplitude(i) for i in range(self.output_dim)]

        output_length_scales = [self._output_length_scales[i] for i in range(self.output_dim)]

        lls_mat = self._left_ls_matrix
        rls_mat = self._right_ls_matrix

        scale_coeff = self._scale_coeff

        if transformed:
            # Forward transform everything
            signal_amplitudes = [sa() for sa in signal_amplitudes]
            noise_amplitudes = [na() for na in noise_amplitudes]
            output_length_scales = [(ols()) for ols in output_length_scales]

            lls_mat = lls_mat()
            rls_mat = rls_mat()

            scale_coeff = scale_coeff()

        return signal_amplitudes, noise_amplitudes, output_length_scales, lls_mat, rls_mat, scale_coeff

    def assign_variables(self, values):
        signal_amplitudes, noise_amplitudes, output_length_scales, lls_mat, rls_mat, scale_coeff = values

        self._left_ls_matrix.assign(lls_mat)
        self._right_ls_matrix.assign(rls_mat)

        self._scale_coeff.assign(scale_coeff)

        for i in range(self.output_dim):
            self._output_length_scales[i].assign(output_length_scales[i])
            self._signal_amplitudes[i].assign(signal_amplitudes[i])
            self._noise_amplitudes[i].assign(noise_amplitudes[i])

    def create_all_hyperparameter_initializers(self, length_scale_init_mode: str, use_gpar_init=False, **kwargs):

        if use_gpar_init:
            print("Fitting GPAR model first!")
            # Initialize by fitting GPAR first
            starting_model = GPARModel(kernel=self.kernel_name,
                                       input_dim=self.input_dim,
                                       output_dim=self.output_dim)

            starting_model = starting_model.condition_on(self.xs, self.ys)

            starting_model.fit(length_scale_init_mode=length_scale_init_mode,
                               fit_joint=False,
                               optimizer_restarts=2)

            print(f"GPAR model fit! Log prob: {starting_model.log_prob(self.xs, self.ys, predictive=False, average=True)}")

            joint_length_scales = starting_model.length_scales()
            signal_amplitudes = starting_model.signal_amplitudes()
            noise_amplitudes = starting_model.noise_amplitudes()

        else:
            # Initialize the hyperparameters for regular GPAR
            all_hyperparams = super().create_all_hyperparameter_initializers(length_scale_init_mode=length_scale_init_mode,
                                                                             **kwargs)

            joint_length_scales, signal_amplitudes, noise_amplitudes = all_hyperparams

        # Separate the input and output length scales
        input_length_scales = []
        input_ls_lower_bounds = []
        input_ls_upper_bounds = []

        output_length_scales = []

        for i in range(self.output_dim):
            joint_ls = joint_length_scales[i]

            # Separate out output length scale
            output_length_scale = BoundedVariable(joint_ls()[self.input_dim:],
                                                  lower=joint_ls.lower[self.input_dim:],
                                                  upper=joint_ls.upper[self.input_dim:],
                                                  dtype=joint_ls.dtype)

            output_length_scales.append(output_length_scale)

            # Separate out input length scale
            input_length_scales.append(joint_ls()[:self.input_dim])
            input_ls_lower_bounds.append(joint_ls.lower[:self.input_dim])
            input_ls_upper_bounds.append(joint_ls.upper[:self.input_dim])

        # Create joint input length scale matrix
        ls_mat = tf.stack(input_length_scales, axis=0)

        self._input_ls_lower_bound.assign(tf.stack(input_ls_lower_bounds, axis=0))
        self._input_ls_upper_bound.assign(tf.stack(input_ls_upper_bounds, axis=0))

        # backward transform the length-scale matrix
        ls_mat = map_from_bounded_interval(ls_mat, lower=self._input_ls_lower_bound, upper=self._input_ls_upper_bound)

        # SVD decomposition of the length scale matrix
        s, u, v = tf.linalg.svd(ls_mat)

        # Create low-rank approximation using Eckart-Young-Mirsky theorem
        s = tf.sqrt(s[:self.latent_dim], tf.float64)
        s_diag = tf.linalg.diag(s)

        # Left length scale initializer: U * sqrt(S)
        lls_init = tf.matmul(u[:, :self.latent_dim], s_diag)

        # Left length scale initializer: sqrt(S) * V^T
        rls_init = tf.matmul(s_diag, tf.transpose(v[:, :self.latent_dim]))

        # The initial scale coefficient is going to be the Frobenius norm of the original matrix
        scale_coeff_init = tf.reduce_sum(ls_mat * ls_mat)

        # Create container for matrix factors:
        # Left length scale (LLS) matrix and Right length scale (RLS) matrix
        lls_mat = UnconstrainedVariable(lls_init, dtype=tf.float64)
        rls_mat = UnconstrainedVariable(rls_init, dtype=tf.float64)

        scale_coeff = PositiveVariable(scale_coeff_init, dtype=tf.float64)

        return lls_mat, rls_mat, scale_coeff, output_length_scales, signal_amplitudes, noise_amplitudes

    def initialize_gp_hyperparameters(self, index, length_scale_init_mode, **kwargs):
        raise ModelError("The hyperparameters of the MF-GPAR model cannot " "be initialized for individual GPs!")

    def initialize_hyperparameters(self, length_scale_init_mode, **kwargs):
        hyperparams = self.create_all_hyperparameter_initializers(length_scale_init_mode=length_scale_init_mode,
                                                                  **kwargs)

        lls_mat, rls_mat, scale_coeff, output_length_scales, signal_amplitudes, noise_amplitudes = hyperparams

        self._left_ls_matrix.assign_var(lls_mat)
        self._right_ls_matrix.assign_var(rls_mat)

        self._scale_coeff.assign_var(scale_coeff)

        for i in range(self.output_dim):
            self._output_length_scales[i].assign_var(output_length_scales[i])
            self._signal_amplitudes[i].assign_var(signal_amplitudes[i])
            self._noise_amplitudes[i].assign_var(noise_amplitudes[i])

        return self._left_ls_matrix, self._right_ls_matrix, self._scale_coeff, self._output_length_scales, self._signal_amplitudes, self._noise_amplitudes

    @staticmethod
    def restore(save_path):

        with open(save_path + ".json", "r") as config_file:
            config = json.load(config_file)

        model = MatrixFactorizedGPARModelV2.from_config(config, )

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
            "verbose": self.verbose,
        }

    @staticmethod
    def from_config(config, **kwargs):
        return MatrixFactorizedGPARModelV2(**config)
