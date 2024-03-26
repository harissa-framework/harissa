import pytest
import sys
import re
from importlib.metadata import version, PackageNotFoundError

import harissa

# https://packaging.python.org/en/latest/specifications/version-specifiers/#public-version-identifiers
version_pattern = re.compile(
    r'^(\d+!)?\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$'
)

@pytest.fixture
def __version__():
    sys_path = sys.path
    sys.path = [harissa.__path__[0]]

    from __init__ import __version__

    yield __version__

    sys.path = sys_path

def test_version():
    assert re.match(version_pattern, harissa.__version__) is not None
    assert harissa.__version__ == version('harissa')

def test_unknown_version(__version__):
    with pytest.raises(PackageNotFoundError):
        version('harissa')

    assert __version__ == 'unknown version'

    

