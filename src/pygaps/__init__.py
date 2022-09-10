# pylint: disable=W0614,W0611,W0622,C0411
# flake8: noqa
# isort:skip_file

__author__ = 'Paul Iacomi'
__docformat__ = 'restructuredtext'

# Solve version shenanigans on conda-build
try:
    # -- Distribution mode --
    # import from _version.py generated by setuptools_scm during release
    from ._version import __version__
except ImportError:
    # -- Source mode --
    # use setuptools_scm to get the current version from src using git
    from setuptools_scm import get_version as _gv
    from os import path as _path
    __version__ = _gv(_path.join(_path.dirname(__file__), _path.pardir))

import sys
from .logging import logger

# This code is written for Python 3.
if sys.version_info[0] != 3:
    logger.error("pyGAPS requires Python 3.")
    sys.exit(1)

# Let users know if they're missing any hard dependencies
hard_dependencies = ("numpy", "pandas", "scipy")
soft_dependencies = {"CoolProp": "Used for many thermodynamic backend calculations."}
missing_dependencies = []

import importlib
for dependency in hard_dependencies:
    if not importlib.util.find_spec(dependency):
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError(f"Missing required dependencies {missing_dependencies}")

for dependency, reason in soft_dependencies.items():
    if not importlib.util.find_spec(dependency):
        logger.warning(f"Missing important package {dependency}. {reason}")

del hard_dependencies, soft_dependencies, dependency, missing_dependencies

# Data
from pygaps.data import DATABASE
from pygaps.data import ADSORBATE_LIST
from pygaps.data import MATERIAL_LIST
from pygaps.data import load_data

# Thermodynamic backend
from pygaps.utilities.coolprop_utilities import thermodynamic_backend
from pygaps.utilities.coolprop_utilities import backend_use_coolprop
from pygaps.utilities.coolprop_utilities import backend_use_refprop

# Core classes
from pygaps.core.adsorbate import Adsorbate
from pygaps.core.material import Material
from pygaps.core.pointisotherm import PointIsotherm
from pygaps.core.modelisotherm import ModelIsotherm

# Data load
load_data()
del load_data

# Other user-facing functions
# from .api import *
