import abc


class AbstractModel(object):
  
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, X, Y, modelParams):
        pass

    @abc.abstractmethod
    def addPoint(self, x, y):
        pass

    @abc.abstractmethod
    def predictBatch(self, X, samples):
        pass


