from .optimizer import Optimizer

_optimizers = {
    'default': Optimizer,
}


class OptimizerLoaderError(Exception):
    def __init__(self, message: str):
        super().__init__("Cannot load optimizer: " + message)


def load_optimizer(config) -> Optimizer:
    try:
        return _optimizers[config['name']](**config.get('parameters', {}))
    except KeyError as e:
        raise OptimizerLoaderError("Key " + str(e) + " not found")
    except Exception as e:
        raise OptimizerLoaderError(str(e))
