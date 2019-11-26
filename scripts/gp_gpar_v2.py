import matplotlib.pyplot as plt
import numpy as np

from boa.models.fully_factorized_gp_v2 import FullyFactorizedGPModel
from boa.models.gpar_v2 import GPARModel


def run():

    # Test function
    def f(x):
        return np.sinc(3 * x[:, 0]).reshape(-1, 1)

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate input data
    x_train = np.random.rand(8, 2) * 2 - 1
    pseudo_point = np.array([[0.8, 0.3]])
    x_train = np.vstack([x_train, pseudo_point])

    y_train = f(x_train)

    # Points for plotting
    x_cont = np.arange(-1.5, 1.5, 0.02).reshape(-1, 1)
    x_cont = np.hstack([x_cont, x_cont])

    # FF-GP model
    ff_gp = FullyFactorizedGPModel(kernel='rbf', num_optimizer_restarts=10, verbose=False)
    ff_gp = ff_gp | (x_train, y_train)
    ff_gp.fit()

    ff_gp.add_pseudo_point(pseudo_point)
    y_pred_ff_gp, var_pred_ff_gp = ff_gp.predict_batch(x_cont)

    y_pred_ff_gp = y_pred_ff_gp.numpy()
    var_pred_ff_gp = var_pred_ff_gp.numpy()

    # GPAR model
    gpar = GPARModel(kernel='rbf', num_optimizer_restarts=10, verbose=False)
    gpar = gpar | (x_train, y_train)
    gpar.fit()

    gpar.add_pseudo_point(pseudo_point)
    y_pred_gpar, var_pred_gpar = gpar.predict_batch(x_cont)

    y_pred_gpar = y_pred_gpar.numpy()
    var_pred_gpar = var_pred_gpar.numpy()

    # Plot results
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

    ax.plot(x_cont[:, 0], f(x_cont), color='black', linestyle='dashed', label='f', zorder=-1)

    ax.plot(x_cont[:, 0], y_pred_ff_gp, color='C0', zorder=-1, label=r'$\mathcal{GP}$')
    ax.fill_between(x_cont.T[0],
                    (y_pred_ff_gp + 2 * np.sqrt(var_pred_ff_gp)).T[0],
                    (y_pred_ff_gp - 2 * np.sqrt(var_pred_ff_gp)).T[0],
                    color='C0',
                    alpha=0.3,
                    zorder=-1,
                    label=r'$\pm2\sigma$')

    ax.plot(x_cont[:, 0], y_pred_gpar, color='C1', zorder=-1, label=r'$\mathcal{GPAR}$', linestyle='dashed')
    ax.fill_between(x_cont.T[0],
                    (y_pred_gpar + 2 * np.sqrt(var_pred_gpar)).T[0],
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

    fig.show()


if __name__ == "__main__":
    run()
