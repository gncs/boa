import json


class ModelParam(object):
    def __init__(self, name, input_):
        self.name = name
        if name == 'dgp':
            self.hidden_layers = 1
            self.Ma = 50
            self.Mb = None
            self.nodes = 1
            self.outputs_per_node = 10
            self.learning_rate = 0.01
            self.mdecay = 5e-2
            self.minibatch_size = 10000
            self.fixed_mean = True
            self.burn_in_iter = 5000
            self.maxiter = 5000
            self.minibatch_size = 10000
            self.epsilon = 0.01
            self.num_samples = 100
            self.spacing_samples = 50
            self.full_input_layer = False
        elif name == 'gp':
            self.kern = 'rbf'
            self.optimizer_restarts = 10
        elif name == 'rng':
            pass
        else:
            raise Exception('Incorrect param name')

        if input_ is not None:
            if isinstance(input, basestring):
                data = json.load(open(input_))
                for key, value in data.iteritems():
                    self.setattr(key, value)
            else:
                for key, value in input_.iteritems():
                    self.__setattr__(key, value)
