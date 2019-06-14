import matplotlib.pyplot as plt
import numpy as np

from boa.models.gp import GPModel
from boa.models.gpar import GPARModel


def f(x):
    return np.sinc(3 * x[:, 0]).reshape(-1, 1)


np.random.seed(42)
X_train = np.random.rand(8, 2) * 2 - 1
pseudo_point = np.array([0.8, 0.3]).reshape(1, -1)
X_train = np.vstack([X_train, pseudo_point])

Y_train = f(X_train)

# Points for plotting
x_cont = np.arange(-1.5, 1.5, 0.02).reshape(-1, 1)
x_cont = np.hstack([x_cont, x_cont])

# GP
model1 = GPModel(kernel='rbf', num_optimizer_restarts=10)
model1.set_data(X_train, Y_train)
model1.train()

model1.add_pseudo_point(pseudo_point)
y_pred_1, var_pred_1 = model1.predict_batch(x_cont)

# GPAR
model2 = GPARModel(kernel='rbf', num_optimizer_restarts=10)
model2.set_data(X_train, Y_train)
model2.train()

model2.add_pseudo_point(pseudo_point)
y_pred_2, var_pred_2 = model2.predict_batch(x_cont)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

ax.plot(x_cont[:, 0], f(x_cont), color='black', linestyle='dashed', label='f', zorder=-1)

ax.plot(x_cont[:, 0], y_pred_1, color='C0', zorder=-1, label=r'$\mathcal{GP}$')
ax.fill_between(x_cont.T[0], (y_pred_1 + 2 * np.sqrt(var_pred_1)).T[0], (y_pred_1 - 2 * np.sqrt(var_pred_1)).T[0],
                color='C0', alpha=0.3, zorder=-1, label=r'$\pm$2$\sigma$')
ax.plot(x_cont[:, 0], y_pred_2, color='C1', zorder=-1, label=r'$\mathcal{GPAR}$', linestyle='dashed')
ax.fill_between(x_cont.T[0], (y_pred_2 + 2 * np.sqrt(var_pred_2)).T[0], (y_pred_2 - 2 * np.sqrt(var_pred_2)).T[0],
                color='C1', alpha=0.3, zorder=-1, label=r'$\pm$2$\sigma$')

# Data points
ax.scatter(x=X_train[:, 0], y=Y_train[:, 0], s=30, c='white', edgecolors='black', label=r'$\mathcal{D}$')
ax.scatter(x=model1.xs[-1:, 0], y=model1.ys[-1:, 0], c='C0', edgecolor='black', label='pseudo point')
ax.scatter(x=model2.xs[-1:, 0], y=model2.ys[-1:, 0], c='C1', edgecolor='black', label='pseudo point')

ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1.0))
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')

fig.show()
