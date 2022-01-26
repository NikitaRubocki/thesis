"""
Programmer: Nikita Rubocki
Company: NIS
Date:  July 2021
Purpose: Interface for creating the ensemble via a self-registering ModelFactory
Sklearn citation: Buitinck et al., 2013
"""

from .model import ModelFactory
from .model import Model
from ._basic import Basic
from ._permutation import Permutation
from ._drop_col import DropCol
from ._error import Error
from ._sequential import Sequential
