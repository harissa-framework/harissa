import pytest
import sys
import re
from importlib.metadata import version, PackageNotFoundError
from importlib import import_module

import harissa

# https://packaging.python.org/en/latest/specifications/version-specifiers/#public-version-identifiers
version_pattern = re.compile(
    r'^(\d+!)?\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$'
)

@pytest.fixture
def __version__():
    sys_path = sys.path
    sys_modules = sys.modules
    sys.path = [harissa.__path__[0]]
    if 'alive_progress.animations.spinners' in sys.modules:
        del sys.modules['alive_progress.animations.spinners']
    if 'alive_progress' in sys.modules:
        del sys.modules['alive_progress']

    mod = import_module('__init__')

    yield mod.__version__

    sys.path = sys_path
    sys.modules = sys_modules

def test_version():
    assert re.match(version_pattern, harissa.__version__) is not None
    assert harissa.__version__ == version('harissa')

def test_unknown_version(__version__):
    with pytest.raises(PackageNotFoundError):
        version('harissa')

    assert __version__ == 'unknown version'
