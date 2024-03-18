import pytest
import sys
from importlib.metadata import version, PackageNotFoundError

import harissa

@pytest.fixture
def simulate_import_without_installing():
    sys_path = sys.path
    sys.path = [harissa.__path__[0]]

    yield

    sys.path = sys_path

def test_version():
    assert harissa.__version__ == version('harissa')

def test_unknown_version(simulate_import_without_installing):
    with pytest.raises(PackageNotFoundError):
        version('harissa')

    from __init__ import __version__

    assert __version__ == 'unknown version'

    

