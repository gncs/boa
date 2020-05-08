import logging
import json

from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import trange

from boa.models.multi_output_gp_regression_model import ModelError
from .gpar import GPARModel

from boa.core.utils import setup_logger, tensor_hash, tf_custom_gradient_method
from not_tf_opt import AbstractVariable, BoundedVariable, UnconstrainedVariable, PositiveVariable
from not_tf_opt import map_to_bounded_interval, map_from_bounded_interval, minimize, get_reparametrizations

from boa import ROOT_DIR

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file=f"{ROOT_DIR}/../logs/mf_gpar.log")

tfl = tf.linalg


__all__ = [
    "SVDFactorizedGPARModel"
]


class SVDFactorizedGPARModel(GPARModel):
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

        super(SVDFactorizedGPARModel, self).__init__(
            kernel=kernel,
            input_dim=input_dim,
            output_dim=output_dim,
            verbose=verbose,
            name=name,
            **kwargs)

        max_latent_dim = min(input_dim, output_dim)
        if latent_dim > max_latent_dim:
            print(f"Latent dimension must be less than min(input_dim, output_dim) = {max_latent_dim}, but was {latent_dim}!")

        self.latent_dim = latent_dim

        # The length scale matrix is assumed to be output_dim x input_dim

        # The product of these gives the log-lengthscales!
        # Note: the left vectors decrease in dimensionality, while the right ones increase. This is intentional,
        # Since in SVD, we have M = USV^T, and the composition of Householder transformations will reflect this structure
        self._left_householder_vectors = [UnconstrainedVariable(tf.ones(i, dtype=tf.float64),
                                                                name=f"left_householder_vector_{i}")
                                          for i in range(self.output_dim,
                                                         self.output_dim - self.latent_dim,
                                                         -1)]

        self._right_householder_vectors = [UnconstrainedVariable(tf.ones(i, dtype=tf.float64),
                                                                 name=f"right_householder_vector_{i}")
                                           for i in range(self.input_dim - self.latent_dim + 1,
                                                          self.input_dim + 1,
                                                          1)]

        self._singular_values_reparam = PositiveVariable(
            tf.ones(self.latent_dim, dtype=tf.float64),
            name="singular_values_reparameterization")

        self._input_ls_lower_bound = tf.Variable(tf.zeros([0, 0], dtype=tf.float64),
                                                 shape=(None, None),
                                                 dtype=tf.float64,
                                                 name="ls_lower_bound",
                                                 trainable=False)

        self._input_ls_upper_bound = tf.Variable(tf.zeros([0, 0], dtype=tf.float64),
                                                 shape=(None, None),
                                                 dtype=tf.float64,
                                                 name="ls_upper_bound",
                                                 trainable=False)

        self._output_length_scales: List[BoundedVariable] = [
            BoundedVariable(tf.zeros([self.gp_input_dim(i) - self.input_dim]),
                            lower=-np.inf,
                            upper=np.inf,
                            name=f"output_length_scales_{i}") for i in range(self.output_dim)
        ]

    @property
    def singular_values(self):
        """
        since self._singular_values_reparam always contains positive entries,
        this will ensure that this result is always is positive descending
        :return:
        """
        return tf.cumsum(self._singular_values_reparam())[::-1]

    def assign_singular_values(self, v):
        """
        v is assumed to be positive descending

        #TODO: Add check
        :param v:
        :return:
        """
        if isinstance(v, AbstractVariable):
            v = v()

        reparam = tf.concat([v[:-1] - v[1:], v[-1:]], axis=0)[::-1]

        self._singular_values_reparam.assign(reparam)

    def householder_reflector(self, v, dim, as_matrix=False):

        if isinstance(v, AbstractVariable):
            v = v()

        v = tf.convert_to_tensor(v, dtype=tf.float64)

        if tf.rank(v) > 1:
            raise ValueError(f"v must be rank 1, but had shape {v.shape}!")

        v_len = v.shape[0]

        if v_len > dim:
            raise ValueError(f"The dimension of v must be less than dim ({dim}), but was {v_len}!")

        reflector = tfl.LinearOperatorBlockDiag([tfl.LinearOperatorIdentity(num_rows=dim - v_len, dtype=v.dtype),
                                                 tfl.LinearOperatorHouseholder(v)])

        if as_matrix:
            reflector = reflector.to_dense()

        return reflector

    def orthogonal_transform(self, vectors, dim, as_matrix=False):

        orth_transform = tfl.LinearOperatorComposition(
            [self.householder_reflector(v, dim, as_matrix=False) for v in vectors])

        if as_matrix:
            orth_transform = orth_transform.to_dense()

        return orth_transform

    @property
    def input_length_scales(self):
        left_orth = self.orthogonal_transform(self._left_householder_vectors, self.output_dim).to_dense()
        right_orth = self.orthogonal_transform(self._right_householder_vectors, self.input_dim).to_dense()

        # Chop of the "unnecessary parts"
        left_orth = left_orth[:, :self.latent_dim]
        right_orth = right_orth[:self.latent_dim, :]

        scaling = tfl.diag(self.singular_values)

        svd = tf.einsum('nd, dd, dm -> nm', left_orth, scaling, right_orth)

        ls_mat = map_to_bounded_interval(svd, lower=self._input_ls_lower_bound, upper=self._input_ls_upper_bound)

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
        print(self.singular_values)

        input_length_scales = self.input_length_scales
        output_length_scales = self._output_length_scales

        s = tf.linalg.svd(input_length_scales, compute_uv=False)
        print(s)

        length_scales = [tf.concat([input_length_scales[i, :], output_length_scales[i]()], axis=0)
                         for i in range(self.output_dim)]

        return list(map(lambda x: (lambda: x), length_scales))

    def gp_variables_to_train(self, index, transformed):
        raise ModelError("SVD-GPAR model cannot be trained in a factorized manner!")

    def gp_assign_variables(self, index, values):
        raise ModelError("Variables for SVD-GPAR model cannot be assigned " "in a factorized manner!")

    def variables_to_train(self, transformed):
        signal_amplitudes = [self.gp_signal_amplitude(i) for i in range(self.output_dim)]

        noise_amplitudes = [self.gp_noise_amplitude(i) for i in range(self.output_dim)]

        output_length_scales = [self._output_length_scales[i] for i in range(self.output_dim)]

        left_householder_vectors = [self._left_householder_vectors[i] for i in range(self.latent_dim)]
        right_householder_vectors = [self._right_householder_vectors[i] for i in range(self.latent_dim)]
        singular_values = self._singular_values_reparam

        if transformed:
            # Forward transform everything
            signal_amplitudes = [sa() for sa in signal_amplitudes]
            noise_amplitudes = [na() for na in noise_amplitudes]
            output_length_scales = [(ols()) for ols in output_length_scales]

            left_householder_vectors = [lhv() for lhv in left_householder_vectors]
            right_householder_vectors = [rhv() for rhv in right_householder_vectors]

            singular_values = self.singular_values

        return (signal_amplitudes,
                noise_amplitudes,
                output_length_scales,
                left_householder_vectors,
                right_householder_vectors,
                singular_values)

    def assign_variables(self, values):

        (signal_amplitudes,
         noise_amplitudes,
         output_length_scales,
         left_householder_vectors,
         right_householder_vectors,
         singular_values) = values

        for i in range(self.output_dim):
            self._output_length_scales[i].assign(output_length_scales[i])
            self._signal_amplitudes[i].assign(signal_amplitudes[i])
            self._noise_amplitudes[i].assign(noise_amplitudes[i])

        for i in range(self.latent_dim):
            self._left_householder_vectors[i].assign(left_householder_vectors[i])
            self._right_householder_vectors[i].assign(right_householder_vectors[i])

        self.assign_singular_values(singular_values)

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
        #ls_mat = tfp.math.softplus_inverse(ls_mat)

        # SVD decomposition of the length scale matrix
        s, u, v = tf.linalg.svd(ls_mat, full_matrices=True)

        singular_values = PositiveVariable(s[:self.latent_dim])

        left_householder_vectors = [UnconstrainedVariable(lhv)
                                    for lhv in find_householder_vectors(u)[::-1][:self.latent_dim]]

        right_householder_vectors = [UnconstrainedVariable(rhv)
                                     for rhv in find_householder_vectors(v)[-self.latent_dim:]]

        return (left_householder_vectors,
                right_householder_vectors,
                singular_values,
                output_length_scales,
                signal_amplitudes,
                noise_amplitudes)

    def initialize_gp_hyperparameters(self, index, length_scale_init_mode, **kwargs):
        raise ModelError("The hyperparameters of the MF-GPAR model cannot " "be initialized for individual GPs!")

    def initialize_hyperparameters(self, length_scale_init_mode, **kwargs):
        hyperparams = self.create_all_hyperparameter_initializers(length_scale_init_mode=length_scale_init_mode,
                                                                  **kwargs)

        (left_householder_vectors,
         right_householder_vectors,
         singular_values,
         output_length_scales,
         signal_amplitudes,
         noise_amplitudes) = hyperparams

        for i in range(self.output_dim):
            self._output_length_scales[i].assign_var(output_length_scales[i])
            self._signal_amplitudes[i].assign_var(signal_amplitudes[i])
            self._noise_amplitudes[i].assign_var(noise_amplitudes[i])

        for i in range(self.latent_dim):
            self._left_householder_vectors[i].assign_var(left_householder_vectors[i])
            self._right_householder_vectors[i].assign_var(right_householder_vectors[i])

        self.assign_singular_values(singular_values)

        return (self._left_householder_vectors,
                self._right_householder_vectors,
                self._singular_values_reparam,
                self._output_length_scales,
                self._signal_amplitudes,
                self._noise_amplitudes)

    @staticmethod
    def restore(save_path):

        with open(save_path + ".json", "r") as config_file:
            config = json.load(config_file)

        model = SVDFactorizedGPARModel.from_config(config, )

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
        return SVDFactorizedGPARModel(**config)


def find_householder_vectors(orthogonal_mat):
    def solve_reflection_vector(orthogonal_mat):
        leading_coeff = tf.math.sqrt((1. - orthogonal_mat[0, 0]) / 2.)
        remaining_coeffs = orthogonal_mat[1:, 0] / (-2. * leading_coeff)
        return tf.concat([[leading_coeff], remaining_coeffs], axis=0)

    n = orthogonal_mat.shape[0]
    dtype = orthogonal_mat.dtype

    if n == 1:
        return [tf.ones(1, dtype=dtype)]

    u = solve_reflection_vector(orthogonal_mat)

    next_block = tf.linalg.solve(tf.eye(n - 1, dtype=dtype) - 2. * u[None, 1:] * u[1:, None],
                                 orthogonal_mat[1:, 1:])

    reflection_vectors = find_householder_vectors(next_block)
    reflection_vectors.append(u)

    return reflection_vectors