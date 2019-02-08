from distutils.util import convert_path
from typing import Dict

from setuptools import setup, find_packages


def readme() -> str:
    with open('README.md') as f:
        return f.read()


version_dict = {}  # type: Dict[str, str]
with open(convert_path('boa/version.py')) as file:
    exec(file.read(), version_dict)

setup(
    name='boa',
    version=version_dict['__version__'],
    description='Multi-Objective Bayesian Optimization Program for the gem5-Aladdin SoC Simulator',
    long_description=readme(),
    classifiers=['Programming Language :: Python :: 3.6'],
    author='Marton Havasi and Gregor Simm',
    author_email='',
    entry_points={
        'console_scripts': ['boa = boa.main:hook'],
    },
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'GPy',
        'matplotlib',  # required by GPy
        'pyyaml',
    ],
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
)
