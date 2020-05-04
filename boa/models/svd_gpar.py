import logging
import json

from typing import List

import numpy as np
import tensorflow as tf

from tqdm import trange

from boa.models.multi_output_gp_regression_model import ModelError
from .gpar import GPARModel

from boa.core.utils import setup_logger, tensor_hash, tf_custom_gradient_method
from not_tf_opt import AbstractVariable, BoundedVariable, UnconstrainedVariable, PositiveVariable
from not_tf_opt import map_to_bounded_interval, map_from_bounded_interval, minimize, get_reparametrizations

from boa import ROOT_DIR

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file=f"{ROOT_DIR}/../logs/mf_gpar.log")

tfl = tf.linalg


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
            _create_length_scales=False,  # Never create the length scales in the parent class
            verbose=verbose,
            name=name,
            **kwargs)

        self.latent_dim = latent_dim

        # The length scale matrix is assumed to be output_dim x input_dim

        # The product of these gives the log-lengthscales!
        # Note: the left vectors decrease in dimensionality, while the right ones increase. This is intentional,
        # Since in SVD, we have M = USV^T, and the composition of Householder transformations will reflect this structure
        self._left_householder_vectors = [UnconstrainedVariable(tf.ones(i, dtype=tf.float64),
                                                                name=f"left_householder_vector_{i}")
                                          for i in range(self.output_dim, 0, -1)]

        self._right_householder_vectors = [UnconstrainedVariable(tf.ones(i, dtype=tf.float64),
                                                                 name=f"right_householder_vector_{i}")
                                           for i in range(1, self.input_dim + 1, 1)]

        self._singular_values = PositiveVariable(
            tf.ones(tf.minimum(self.input_dim, self.output_dim), dtype=tf.float64),
            name="singular_values")

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

        self._cached_input_length_scales = tf.Variable(tf.zeros([output_dim, input_dim], dtype=tf.float64),
                                                       trainable=False,
                                                       name="cached_input_length_scales")

        self._cached_lhv_jacobians = {lhv.var.name: tf.Variable(tf.zeros((output_dim, input_dim) + lhv.shape,
                                                                     dtype=tf.float64),
                                                            name=f"cached_lhv_{i}_jacobian",
                                                            trainable=False)
                                      for i, lhv in enumerate(self._left_householder_vectors)}

        self._cached_rhv_jacobians = {rhv.var.name: tf.Variable(tf.zeros((output_dim, input_dim) + rhv.shape,
                                                                     dtype=tf.float64),
                                                            name=f"cached_rhv_{i}_jacobian",
                                                            trainable=False)
                                      for i, rhv in enumerate(self._right_householder_vectors)}

        self._cached_sv_jacobians = tf.Variable(tf.ones((output_dim, input_dim) + self._singular_values.shape,
                                                        dtype=tf.float64),
                                                name="cached_singular_values_jacobian")

        self._input_length_scale_hash = tf.Variable(42, dtype=tf.int64, name="input_length_scale_hash", trainable=False)

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
        left_orth = self.orthogonal_transform(self._left_householder_vectors, self.output_dim)
        right_orth = self.orthogonal_transform(self._right_householder_vectors, self.input_dim)
        scaling = tfl.LinearOperatorDiag(self._singular_values())

        # We need to chop one of the orthogonal matrices
        if self.output_dim > self.input_dim:
            scaling_times_right_orth = tfl.LinearOperatorComposition([scaling, right_orth]).to_dense()
            left_orth = left_orth.to_dense()[:, :self.input_dim]

            svd = tf.matmul(left_orth, scaling_times_right_orth)

        else:
            left_orth_times_scaling = tfl.LinearOperatorComposition([left_orth, scaling]).to_dense()
            right_orth = right_orth.to_dense()[:self.output_dim, :]

            svd = tf.matmul(left_orth_times_scaling, right_orth)

        ls_mat = map_to_bounded_interval(svd, lower=self._input_ls_lower_bound, upper=self._input_ls_upper_bound)

        return ls_mat

    @property
    @tf_custom_gradient_method
    def _input_length_scales(self):

        input_scale_variables = get_reparametrizations([self._right_householder_vectors,
                                                        self._left_householder_vectors,
                                                        self._singular_values])

        # Determine if there were any changes, i.e. whether the length scales should be recomputed
        ils_hash = tensor_hash(input_scale_variables)

        if ils_hash != int(self._input_length_scale_hash):

            print("Recalculating Jacobians!")
            self._input_length_scale_hash.assign(ils_hash)

            with tf.GradientTape() as tape:
                ils = self._input_length_scales()

            ils_jac = tape.jacobian(ils, input_scale_variables)

            assert len(ils_jac[0]) == len(self._right_householder_vectors), len(ils_jac[0])
            assert len(ils_jac[1]) == len(self._left_householder_vectors), len(ils_jac[1])
            assert ils_jac[2].shape == self._cached_sv_jacobians.shape

            for i in range(len(ils_jac[0])):
                self._cached_rhv_jacobians[self._right_householder_vectors[i].var.name].assign(ils_jac[0][i])

            for i in range(len(ils_jac[1])):
                self._cached_lhv_jacobians[self._left_householder_vectors[i].var.name].assign(ils_jac[1][i])

            self._cached_sv_jacobians.assign(ils_jac[2])

            self._cached_input_length_scales.assign(ils)

        else:
            print("Reusing gradients")

        def grad(dy, variables):

            grads = []

            for variable in variables:

                # Calculate appropriate Vector - Jacobian products
                if variable.name in self._cached_lhv_jacobians:
                    vjp = tf.einsum('ij, ijk -> k', dy, self._cached_lhv_jacobians[variable.name])

                elif variable.name in self._cached_rhv_jacobians:
                    vjp = tf.einsum('ij, ijk -> k', dy, self._cached_rhv_jacobians[variable.name])

                elif variable.name == self._singular_values.var.name:
                    vjp = tf.einsum('ij, ijk -> k', dy, self._cached_sv_jacobians)

                else:
                    print(variable.name)
                    vjp = None

                grads.append(vjp)

            return (), grads

        return self._cached_input_length_scales.value(), grad

    def has_explicit_length_scales(self):
        return False

    def length_scales(self, index):
        # Wrap in a lambda so it imitates the forward transform of a variable
        return lambda: tf.concat([self.input_length_scales[index, :], self._output_length_scales[index]()], axis=0)

    def gp_variables_to_train(self, index, transformed):
        raise ModelError("MF-GPAR model cannot be trained in a factorized manner!")

    def gp_assign_variables(self, index, values):
        raise ModelError("Variables for MF-GPAR model cannot be assigned " "in a factorized manner!")

    def variables_to_train(self, transformed):
        signal_amplitudes = [self.signal_amplitude(i) for i in range(self.output_dim)]

        noise_amplitudes = [self.noise_amplitude(i) for i in range(self.output_dim)]

        output_length_scales = [self._output_length_scales[i] for i in range(self.output_dim)]

        left_householder_vectors = [self._left_householder_vectors[i] for i in range(self.output_dim)]
        right_householder_vectors = [self._right_householder_vectors[i] for i in range(self.input_dim)]
        singular_values = self._singular_values

        if transformed:
            # Forward transform everything
            signal_amplitudes = [sa() for sa in signal_amplitudes]
            noise_amplitudes = [na() for na in noise_amplitudes]
            output_length_scales = [(ols()) for ols in output_length_scales]

            left_householder_vectors = [lhv() for lhv in left_householder_vectors]
            right_householder_vectors = [rhv() for rhv in right_householder_vectors]

            singular_values = singular_values()

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

            self._left_householder_vectors[i].assign(left_householder_vectors[i])

        for i in range(self.input_dim):
            self._right_householder_vectors[i].assign(right_householder_vectors[i])

        self._singular_values.assign(singular_values)

    def create_all_hyperparameter_initializers(self, length_scale_init_mode: str, **kwargs):

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

        left_householder_vectors = [UnconstrainedVariable(tf.random.normal(shape=(i,), dtype=tf.float64))
                                    for i in range(self.output_dim, 0, -1)]

        right_householder_vectors = [UnconstrainedVariable(tf.random.normal(shape=(i,), dtype=tf.float64))
                                     for i in range(1, self.input_dim + 1, 1)]

        # backward transform the length-scale matrix
        ls_mat = map_from_bounded_interval(ls_mat, lower=self._input_ls_lower_bound, upper=self._input_ls_upper_bound)

        # SVD decomposition of the length scale matrix
        s, u, v = tf.linalg.svd(ls_mat, full_matrices=True)

        singular_values = PositiveVariable(s)

        # Minimize the Frobenius norm between the components of the SVD and our orthogonal matrices
        def frobenius_loss(a, b):
            return tf.reduce_sum(tf.math.squared_difference(a, b))

        left_loss = lambda vectors: frobenius_loss(u, self.orthogonal_transform(vectors, dim=self.output_dim,
                                                                                as_matrix=True))
        right_loss = lambda vectors: frobenius_loss(v, self.orthogonal_transform(vectors, dim=self.input_dim,
                                                                                 as_matrix=True))

        res, converged, diverged = minimize(left_loss,
                                            vs=[left_householder_vectors])

        print(f"Left res: {res}, conv: {converged}, div: {diverged}, U frobenius norm: {tf.reduce_sum(u * u)}")

        res, converged, diverged = minimize(right_loss,
                                            vs=[right_householder_vectors])

        print(f"Right res: {res}, conv: {converged}, div: {diverged}, V frobenius norm: {tf.reduce_sum(v * v)}")

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

            self._left_householder_vectors[i].assign_var(left_householder_vectors[i])

        for i in range(self.input_dim):
            self._right_householder_vectors[i].assign_var(right_householder_vectors[i])

        self._singular_values.assign_var(singular_values)

        return (self._left_householder_vectors,
                self._right_householder_vectors,
                self._singular_values,
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
