# import os
# import sys
# from importlib import import_module
# import doctest
# import pytest
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

# @pytest.mark.parametrize('modname', ['armageddon.locator'])
# def test_docstrings(modname):
#     module = import_module(modname)
#     assert doctest.testmod(module).failed == 0

# @pytest.mark.parametrize('modname', ['armageddon.damage'])
# def test_docstrings(modname):
#     module = import_module(modname)
#     assert doctest.testmod(module).failed == 0

import os
import sys
from importlib import import_module
import doctest
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


function = import_module('armageddon')


def test_docstrings():
    test_damage = doctest.testmod(m = function.damage)

    assert test_damage.failed == 0

    test_locator = doctest.testmod(m = function.locator)

    assert test_locator.failed == 0