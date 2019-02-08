import yaml


class ConfigError(Exception):
    """ Error raised by this module """


def read(path: str) -> dict:
    """ Read configuration file. """
    try:
        with open(path, 'r') as f:
            return yaml.load(f)

    except FileNotFoundError:
        raise ConfigError('Configuration file ' + str(path) + ' not found')

    except OSError as e:
        raise ConfigError('Could not read configuration file ' + str(path) + ': ' + str(e))

    except yaml.YAMLError as e:
        raise ConfigError('Could not decode configuration file ' + str(path) + ': ' + str(e))
