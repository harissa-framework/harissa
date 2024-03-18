import pytest
import sys
from importlib.metadata import version, PackageNotFoundError

import harissa

@pytest.fixture
def __version__():
    sys_path = sys.path
    sys.path = [harissa.__path__[0]]

    from __init__ import __version__

    yield __version__

    sys.path = sys_path

def test_version():
    assert harissa.__version__ == version('harissa')

def test_unknown_version(__version__):
    with pytest.raises(PackageNotFoundError):
        version('harissa')

    assert __version__ == 'unknown version'

    

