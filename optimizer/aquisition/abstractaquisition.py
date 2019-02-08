import abc


class AbstractAquisition(object):
  
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, aquisition_params, reference_point):
        pass

    @abc.abstractmethod
    def getAquisitionBatch(self, X, model, frontier):
        pass

    