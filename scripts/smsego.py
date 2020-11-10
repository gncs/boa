import logging

from typing import List

from boa.models.fully_factorized_gp import FullyFactorizedGPModel
from boa.models.bnn.multi_output_bnn import BasicBNN
from boa.acquisition.smsego import SMSEGO
from boa.objective.abstract import AbstractObjective
from boa.optimization.optimizer import Optimizer

from boa.core.utils import setup_logger

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')
logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/smsego_v2.log")

FIG_SAVE_FOLDER = "plots/script_plots/smsego"


def plot(xs, fs, preds, var, acqs, points_xs, points_ys):

    sns.set_style('whitegrid')

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 5))
    ax1.grid(False)
    ax2.grid(False)

    # Solid line
    ax1.plot(xs, fs, color='black', linestyle='dashed', label='f', zorder=-1)
    ax1.plot(xs, preds, color='black', zorder=-1, label=r'$\mathcal{GP}$')
    ax1.fill_between(xs.T[0], (preds + np.sqrt(var)).T[0], (preds - np.sqrt(var)).T[0],
                     color='blue',
                     alpha=0.3,
                     zorder=-1,
                     label=r'$\pm\sigma$')

    # Training set
    ax1.scatter(x=points_xs,
                y=points_ys,
                s=30,
                c='white',
                edgecolors='black',
                label=r'$\{X\}_{N={' + str(points_xs.shape[0]) + '}}$')
    ax1.legend()

    # Acquisition
    ax2.plot(xs, acqs, color='red', label='$a(x)$', zorder=1)
    ax2.axhline(linestyle='dashed', color='black', zorder=-1)

    ax2.set_xlabel('$X$')
    ax2.set_ylabel('$a(X)$')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    return fig


def f(x):
    """
    Target function (noise free).
    """
    return (np.sinc(3 * x) + 0.5 * (x - 0.5)**2).reshape(-1, 1)


def manual_optimization(x_train, y_train, model_type="gp"):


    if model_type == "gp":
        model = FullyFactorizedGPModel(kernel="rbf", input_dim=1, output_dim=1, verbose=False)
    elif model_type == "bnn":
        model = BasicBNN(input_dim=1, output_dim=1)
    else:
        raise NotImplementedError

    # Optimize the hyperparameters
    model = model.condition_on(x_train, y_train)

    if model_type == "gp":
        model.fit(optimizer_restarts=5)
    elif model_type == "bnn":
        model.fit(num_samples=50,
                  burnin=2000,
                  keep_every=50)
    else:
        raise NotImplementedError

    # Set up the acquisition function
    acq = SMSEGO(gain=1., epsilon=0.1)

    # Make predictions and evaluate acquisition function
    x_cont = np.linspace(start=-1.5, stop=1.5, num=200).reshape([-1, 1])

    data_x = x_train.copy()
    data_y = y_train.copy()

    # Perform optimization without the optimizer interface
    for i in range(5):
        y_cont, var_cont = model.predict(x_cont)

        acquisition_values, pred_means = acq.evaluate(model=model,
                                                      xs=data_x,
                                                      ys=data_y,
                                                      candidate_xs=x_cont)

        eval_point = x_cont[np.argmax(acquisition_values)]

        fig = plot(xs=x_cont,
                   fs=f(x_cont),
                   preds=y_cont.numpy(),
                   var=var_cont.numpy(),
                   acqs=acquisition_values,
                   points_xs=data_x,
                   points_ys=data_y)

        fig_path = FIG_SAVE_FOLDER + f"/{model_type}_manual_{i}.pdf"
        fig.savefig(fig_path)
        print(f"Saved image to {fig_path}!")

        # Evaluate function at chosen points
        inp = eval_point
        outp = f(eval_point)

        # Add evaluations to data set and model
        data_x = np.vstack((data_x, inp))
        data_y = np.vstack((data_y, outp))

        model = model.condition_on(inp.reshape([-1, 1]), outp)

        if model_type == "gp":
            model.fit(optimizer_restarts=3)
        elif model_type == "bnn":
            model.fit(num_samples=50,
                      burnin=2000,
                      keep_every=50)
        else:
            raise NotImplementedError


def automated_optimization(x_train, y_train):
    # SMS-EGO
    # Wrap the objective f

    # Make predictions and evaluate acquisition function
    x_cont = np.linspace(start=-1.5, stop=1.5, num=200).reshape([-1, 1])

    class Objective(AbstractObjective):
        def __init__(self, fun, candidates):
            super().__init__()
            self.f = fun
            self.candidates = candidates

        def get_candidates(self) -> np.ndarray:
            return self.candidates

        def get_input_labels(self) -> List[str]:
            return ['X']

        def get_output_labels(self) -> List[str]:
            return ['Y']

        def __call__(self, candidate: np.ndarray) -> np.ndarray:
            return self.f(candidate)

    objective = Objective(fun=f, candidates=x_cont)

    # Set up GP model and train it
    model = FullyFactorizedGPModel(kernel='rbf', input_dim=1, output_dim=1, verbose=False)
    model = model.condition_on(x_train, y_train)
    model.fit(optimizer_restarts=3)

    # Setup the acquisition fn
    acq = SMSEGO(gain=1, epsilon=0.1, reference=[2])

    # Set up the optimizer
    optimizer = Optimizer(max_num_iterations=12, batch_size=1, strict=True)

    data_x, data_y = optimizer.optimize(f=objective,
                                        model=model,
                                        acq_fun=acq,
                                        xs=x_train,
                                        ys=y_train,
                                        candidate_xs=x_cont,
                                        initialization='l2_median',
                                        marginalize_hyperparameters=False,
                                        map_estimate=False,
                                        iters=200,
                                        fit_joint=False,
                                        model_optimizer='l-bfgs-b',
                                        optimizer_restarts=3)

    model = model.condition_on(data_x, data_y, keep_previous=False)

    y_cont, var_cont = model.predict(x_cont)
    acquisition_values, _ = acq.evaluate(
        model=model,
        xs=data_x,
        ys=data_y,
        candidate_xs=x_cont,
        marginalize_hyperparameters=False,
    )

    fig = plot(xs=x_cont,
               fs=f(x_cont),
               preds=y_cont.numpy(),
               var=var_cont.numpy(),
               acqs=acquisition_values,
               points_xs=data_x,
               points_ys=data_y)

    fig_path = FIG_SAVE_FOLDER + "/automated.png"
    fig.savefig(fig_path)
    print(f"Saved image to {fig_path}")


if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate X's and Y's for training.
    x_train = np.array([-0.25, 0, 0.1]).reshape(-1, 1)
    y_train = f(x_train)

    manual_optimization(x_train, y_train, model_type="bnn")

    #automated_optimization(x_train, y_train)
