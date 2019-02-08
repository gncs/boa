import json


class AquisitionParam(object):
    def __init__(self, name, input_):
        self.name = name
        if name == 'smsego':
            self.gain = 2.0
            self.epsilon = 0.001
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