import argparse
import re

from boa.acquisition.loader import load_acquisition
from boa.config import read as read_config
from boa.data import FileHandler, get_init_data
from boa.models.loader import load_model
from boa.objective.loader import load_objective
from boa.optimization.loader import load_optimizer
from boa.util import print_message
from boa.version import __version__


class BOAException(Exception):
    """Exception thrown if version of program and the one specified in configuration file do not match"""


class VersionException(BOAException):
    """Exception thrown if version of program and the one specified in configuration file do not match"""


def check_version(config: dict):
    version_re = re.compile(r'(?P<major>\d+)\.(?P<minor>\d+)(.(?P<path>\d+))?')

    config_match = version_re.match(config['version'])
    program_match = version_re.match(__version__)

    if not config_match or not program_match:
        raise VersionException('Cannot parse version number')

    try:
        if config_match.group('major') != program_match.group('major') or config_match.group(
                'minor') != program_match.group('minor'):
            raise VersionException('Program version ({0}) does not match version in configuration file ({1})'.format(
                __version__, config['version']))
    except AttributeError:
        raise VersionException('Cannot parse version number')


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='boa',
        description='Multi-Objective Bayesian Optimization Program for the gem5-Aladdin SoC Simulator',
    )
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
    parser.add_argument('--config', help='configuration file', required=True)

    return parser


def hook() -> None:
    args = create_parser().parse_args()
    config = read_config(args.config)

    check_version(config)

    print_message('Task: {}'.format(config['task']))

    try:
        model = load_model(config['model'])
        acq = load_acquisition(config['acquisition'])
        optimizer = load_optimizer(config['optimizer'])
        file_handler = FileHandler(config['task'] + '.json')

        objective = load_objective(config['objective'])
        candidate_xs = objective.get_candidates()

        data = get_init_data(config=config['data'], objective=objective, handler=file_handler)
        model.set_data(xs=data.input, ys=data.output)
        model.train()

        xs, ys = optimizer.optimize(
            f=objective,
            model=model,
            acq_fun=acq,
            xs=data.input,
            ys=data.output,
            candidate_xs=candidate_xs,
        )

        data.input = xs
        data.output = ys
        file_handler.save(data)

    except KeyError as e:
        raise BOAException('Key ' + str(e) + ' not found')

    print_message('Done')


if __name__ == '__main__':
    hook()
