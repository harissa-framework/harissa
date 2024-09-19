import sys
from importlib import import_module
from harissa.utils.progress_bar import alive_bar, _IdleBarHandle
from alive_progress.core.progress import __AliveBarHandle
import harissa

def test_alive_bar():
    with alive_bar(1) as bar:
        assert isinstance(bar, __AliveBarHandle)
        bar.text('hello')
        bar.text = 'hello'
        bar()

def test_disable_bar():
    with alive_bar(1, disable=True) as bar:
        assert isinstance(bar, _IdleBarHandle)
        bar.text('hello')
        bar.text = 'hello'
        bar()

def test_error_import():
    sys_modules = sys.modules
    sys_path = sys.path

    sys.path = [harissa.__path__[0]]
    if 'alive_progress' in sys.modules:
        del sys.modules['alive_progress']

    if 'harissa.utils.progress_bar' in sys.modules:
        del sys.modules['harissa.utils.progress_bar']

    mod = import_module('harissa.utils.progress_bar')
    with mod.alive_bar() as bar:
        assert isinstance(bar, mod._IdleBarHandle)

    sys.modules = sys_modules
    sys.path = sys_path
