import os
import sys
from importlib import import_module
import doctest
import pytest
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


func = import_module('armageddon.locator')

@pytest.mark.parametrize('modname', ['armageddon.locator'])
def test_docstrings(modname):
    module = import_module(modname)
    assert doctest.testmod(module).failed == 0
    # assert test.failed == 0