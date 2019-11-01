from typing import List

from boa.models.gp_model_v2 import GPModel
from boa.acquisition.smsego_v2 import SMSEGO
from boa.objective.abstract import AbstractObjective
from boa.optimization.optimizer_v2 import Optimizer

import numpy as np
import matplotlib.pyplot as plt


def plot(xs, fs, preds, var, acqs, points_xs, points_ys):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 5))

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

    fig.subplots_adjust(hspace=0)

    return fig


def run():
    """
    Main function
    """

    # Target function (noise free).
    def f(x):
        return (np.sinc(3 * x) + 0.5 * (x - 0.5) ** 2).reshape(-1, 1)

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate X's and Y's for training.
    x_train = np.array([-0.25, 0, 0.1]).reshape(-1, 1)
    y_train = f(x_train)

    # Infer GP
    model = GPModel(kernel="rbf", num_optimizer_restarts=5)
    model = model | (x_train, y_train)

    # Optimize the hyperparameters
    model.train()

    # Set up the acquisition function
    acq = SMSEGO(gain=1., epsilon=0.1, reference=[2])

    # Make predictions and evaluate acquisition function
    x_cont = np.linspace(start=-1.5, stop=1.5, num=200).reshape([-1, 1])

    data_x = x_train.copy()
    data_y = y_train.copy()

    # Perform optimization without the optimizer interface
    for i in range(5):

        y_cont, var_cont = model.predict_batch(x_cont)

        print(data_x.shape, data_y.shape, x_cont.shape, y_cont.shape, var_cont.shape)

        acquisition_values = acq.evaluate(model=model, xs=data_x, ys=data_y, candidate_xs=x_cont)

        eval_point = x_cont[np.argmax(acquisition_values)]

        fig = plot(
                xs=x_cont,
                fs=f(x_cont),
                preds=y_cont.numpy(),
                var=var_cont.numpy(),
                acqs=acquisition_values,
                points_xs=data_x,
                points_ys=data_y)

        fig.show()

        # Evaluate function at chosen points
        inp = eval_point
        outp = f(eval_point)

        # Add evaluations to data set and model
        data_x = np.vstack((data_x, inp))
        data_y = np.vstack((data_y, outp))
        model.add_true_point(inp.reshape([-1, 1]), outp)

        model.train()


if __name__ == "__main__":
    run()
