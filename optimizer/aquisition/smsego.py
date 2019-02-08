import numpy as np

from optimizer.aquisition.abstractaquisition import AbstractAquisition
from optimizer.optimizer_helper import get_hypervolume, shift


class SMSego(AbstractAquisition):
    def __init__(self, aquisition_params, reference_point, mean, std):
        self.gain = aquisition_params.gain
        self.epsilon = aquisition_params.epsilon
        self.n_dim = reference_point.shape[0]
        self.reference_point = reference_point
	self.mean, self.std = mean, std

    def get_hypervolume(self, frontier, reference_point):
	return get_hypervolume(shift(frontier, self.mean, self.std), reference_point)

    def getAquisitionBatch(self, X, model, frontier):
        n_points = X.shape[0]

        means, vars = model.predictBatch(X, samples=1)
        means = means[0, :, :]
        vars = vars[0, :, :]

        pot_sol = means - self.gain * np.sqrt(np.maximum(vars, 1e-10))
        
        hv_frontier = self.get_hypervolume(frontier, self.reference_point)
        aquisitions = np.ones((n_points))
        for i in range(0, n_points):
            penalty = 0.0
            for k in range(0, frontier.shape[0]):
                if np.all(frontier[k, :] <= pot_sol[i, :] + self.epsilon):
                    p = -1 + np.prod(1 + np.maximum(pot_sol[i, :] - frontier[k, :],
                                     np.zeros_like(pot_sol[i, :])))
                    penalty = np.maximum(penalty, p)

            if penalty == 0.0:
                hv_pot = self.get_hypervolume(np.vstack((pot_sol[i, :], frontier)), self.reference_point)
                aquisitions[i] = -hv_frontier + hv_pot
            else:
                aquisitions[i] = -penalty
        
        return aquisitions


