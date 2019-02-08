import numpy as np

from models.abstractmodel import AbstractModel


class RandomModel(AbstractModel):
    def __init__(self, X, Y, model_params):
        self.dim = Y.shape[1]
        
    def addPoint(self, x, y):        
        pass
        
    def predictBatch(self, X, samples):
        return np.random.rand(samples, X.shape[0], self.dim), np.random.rand(samples, X.shape[0], self.dim)
