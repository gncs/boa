import importlib.util
import os
from typing import Any

from .abstract import AbstractObjective


class ModuleLoadError(Exception):
    """ Error raised when loading the order module fails """


def load_objective(config):
    try:
        module = _load_order_module(config['name'])

        if not issubclass(module.Objective, AbstractObjective):
            raise ModuleLoadError("Objective is not a subclass of 'AbstractObjective'")

        return module.Objective(**config['parameters'])

    except AttributeError as e:
        raise ModuleLoadError(str(e))
    except KeyError as e:
        raise ModuleLoadError('Cannot find key ' + str(e) + '.')


def _load_order_module(module_path) -> Any:
    module_name = os.path.splitext(module_path)[0]
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        m = importlib.util.module_from_spec(spec)

        if not spec.loader:
            raise ModuleLoadError('Cannot load namespace package')

        spec.loader.exec_module(m)
        return m

    except Exception as e:
        raise ModuleLoadError('Cannot load module ' + str(module_path) + ': ' + str(e))
