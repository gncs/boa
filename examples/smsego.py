from typing import List

import matplotlib.pyplot as plt
import numpy as np

from boa.acquisition.smsego import SMSEGO
from boa.models.gp import GPModel
from boa.objective.abstract import AbstractObjective
from boa.optimization.optimizer import Optimizer


def plot(xs, fs, preds, var, acqs, points_xs, points_ys):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 5))

    # Solid line
    ax1.plot(xs, fs, color='black', linestyle='dashed', label='f', zorder=-1)
    ax1.plot(xs, preds, color='black', zorder=-1, label=r'$\mathcal{GP}$')
    ax1.fill_between(xs.T[0], (preds + np.sqrt(var)).T[0], (preds - np.sqrt(var)).T[0], color='blue', alpha=0.3,
                     zorder=-1, label=r'$\pm\sigma$')

    # Training set
    ax1.scatter(x=points_xs, y=points_ys, s=30, c='white', edgecolors='black',
                label=r'$\{X\}_{N={' + str(points_xs.shape[0]) + '}}$')
    ax1.legend()

    # Acquisition
    ax2.plot(xs, acqs, color='red', label='$a(x)$', zorder=1)
    ax2.axhline(linestyle='dashed', color='black', zorder=-1)

    ax2.set_xlabel('$X$')
    ax2.set_ylabel('$a(X)$')

    fig.subplots_adjust(hspace=0)

    return fig


# Target function (noise free).
def f(X):
    return (np.sinc(3 * X) + 0.5 * (X - 0.5) ** 2).reshape(-1, 1)


# Generate X's and Y's for training.
np.random.seed(42)
X_train = np.array([-0.25, 0, 0.1, ]).reshape(-1, 1)
Y_train = f(X_train)

# Setup GP model and train.
model = GPModel(kernel='rbf', num_optimizer_restarts=3)
model.set_data(X_train, Y_train)
model.train()

# Setup acquisition function
acq = SMSEGO(gain=1, epsilon=0.1, reference=[2])

# Make predictions and evaluate acquisition function
x_cont = np.linspace(start=-1.5, stop=1.5, num=200).reshape(-1, 1)

data_x = X_train.copy()
data_y = Y_train.copy()

# Manual exploration
for iteration in range(5):
    y_cont, var_cont = model.predict_batch(x_cont)
    acquisition_values = acq.evaluate(model=model, xs=data_x, ys=data_y, candidate_xs=x_cont)
    eval_point = x_cont[np.argmax(acquisition_values)]

    fig = plot(
        xs=x_cont,
        fs=f(x_cont),
        preds=y_cont,
        var=var_cont,
        acqs=acquisition_values,
        points_xs=data_x,
        points_ys=data_y,
    )
    fig.show()

    # Evaluate function at chosen points
    inp = eval_point
    outp = f(eval_point)

    # Add evaluations to data set and model
    data_x = np.vstack((data_x, inp))
    data_y = np.vstack((data_y, outp))
    model.add_true_point(inp, outp)

    model.train()


# SMS-EGO
# Wrap objective function f
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

# Setup GP model and train.
model = GPModel(kernel='rbf', num_optimizer_restarts=3)
model.set_data(X_train, Y_train)
model.train()

# Setup acquisition function
acq = SMSEGO(gain=1, epsilon=0.1, reference=[2])

# Optimize
optimizer = Optimizer(max_num_iterations=4, batch_size=1)

x_cont = np.linspace(start=-1.5, stop=1.5, num=200).reshape(-1, 1)

data_x, data_y = optimizer.optimize(
    f=objective,
    model=model,
    acq_fun=acq,
    xs=X_train,
    ys=Y_train,
    candidate_xs=x_cont,
)

y_cont, var_cont = model.predict_batch(x_cont)
acquisition_values = acq.evaluate(model=model, xs=data_x, ys=data_y, candidate_xs=x_cont)

fig = plot(
    xs=x_cont,
    fs=f(x_cont),
    preds=y_cont,
    var=var_cont,
    acqs=acquisition_values,
    points_xs=data_x,
    points_ys=data_y,
)
fig.show()
