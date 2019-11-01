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
    author='Machine Learning Group Cambridge',
    author_email='',
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'GPy',
        'matplotlib',  # required by GPy
        'pandas',
        'scikit-learn',
        'tensorflow',
        'stheno',
        'varz'
    ],
    zip_safe=False,
)
