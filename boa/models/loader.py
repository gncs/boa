from .gp import GPModel
from .random import RandomModel

_models = {
    'random': RandomModel,
    'gp': GPModel,
}


class ModelLoaderError(Exception):
    def __init__(self, message: str):
        super().__init__("Cannot load model: " + message)


def load_model(config: dict):
    try:
        return _models[config['name']](**config.get('parameters', {}))
    except KeyError as e:
        raise ModelLoaderError("Key " + str(e) + " not found")
    except Exception as e:
        raise ModelLoaderError(str(e))
