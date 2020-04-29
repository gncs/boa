import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from boa.models.fully_factorized_gp import FullyFactorizedGPModel
from boa.models.gpar import GPARModel

# Always use CPU
tf.config.experimental.set_visible_devices([], 'GPU')

# Set seed for reproducibility
np.random.seed(60)
tf.random.set_seed(60)


def run():

    # Test function
    def f(x):
        return np.sinc(3 * x[:, 0]).reshape(-1, 1)

    # Generate input data
    x_train = np.random.rand(8, 2) * 2 - 1
    pseudo_point = np.array([[0.8, 0.3]])
    x_train = np.vstack([x_train, pseudo_point])

    y_train = f(x_train)

    # Points for plotting
    x_cont = np.arange(-1.5, 1.5, 0.02).reshape(-1, 1)
    x_cont = np.hstack([x_cont, x_cont])

    # FF-GP model
    ff_gp = FullyFactorizedGPModel(kernel='rbf', input_dim=2, output_dim=1, verbose=False)
    ff_gp = ff_gp.condition_on(x_train, y_train)
    ff_gp.fit(optimizer_restarts=10)

    # Add a "pseudo point"
    pred_point, _ = ff_gp.predict(pseudo_point)
    ff_gp = ff_gp.condition_on(pseudo_point, pred_point, keep_previous=True)
    y_pred_ff_gp, var_pred_ff_gp = ff_gp.predict(x_cont, numpy=True)

    # GPAR model
    gpar = GPARModel(kernel='rbf', input_dim=2, output_dim=1, verbose=False)
    gpar = gpar.condition_on(x_train, y_train)
    gpar.fit(optimizer_restarts=10)

    pred_point, _ = gpar.predict(pseudo_point)
    gpar.condition_on(pseudo_point, pred_point, keep_previous=True)
    y_pred_gpar, var_pred_gpar = gpar.predict(x_cont, numpy=True)

    # Plot results
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

    ax.plot(x_cont[:, 0], f(x_cont), color='black', linestyle='dashed', label='f', zorder=-1)

    ax.plot(x_cont[:, 0], y_pred_ff_gp, color='C0', zorder=-1, label=r'$\mathcal{GP}$')
    ax.fill_between(x_cont.T[0], (y_pred_ff_gp + 2 * np.sqrt(var_pred_ff_gp)).T[0],
                    (y_pred_ff_gp - 2 * np.sqrt(var_pred_ff_gp)).T[0],
                    color='C0',
                    alpha=0.3,
                    zorder=-1,
                    label=r'$\pm2\sigma$')

    ax.plot(x_cont[:, 0], y_pred_gpar, color='C1', zorder=-1, label=r'$\mathcal{GPAR}$', linestyle='dashed')
    ax.fill_between(x_cont.T[0], (y_pred_gpar + 2 * np.sqrt(var_pred_gpar)).T[0],
                    (y_pred_gpar - 2 * np.sqrt(var_pred_gpar)).T[0],
                    color='C1',
                    alpha=0.3,
                    zorder=-1,
                    label=r'$\pm2\sigma$')

    # Data points
    ax.scatter(x=x_train[:, 0], y=y_train[:, 0], s=30, c='white', edgecolors='black', label=r'$\mathcal{D}$')
    ax.scatter(x=ff_gp.xs[-1:, 0], y=ff_gp.ys[-1:, 0], c='C0', edgecolors='black', label='pseudo point')
    ax.scatter(x=gpar.xs[-1:, 0], y=gpar.ys[-1:, 0], c='C1', edgecolors='black', label='pseudo point')

    ax.legend(loc='upper left')
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')

    fig.savefig("plots/script_plots/gp_gpar_v2.png")
    print("Figure saved!")


if __name__ == "__main__":
    run()
