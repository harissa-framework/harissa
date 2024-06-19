from contextlib import contextmanager
from typing import Optional, Any


class _IdleBarHandle:
    text = None
    
    def text(self, text: str):
        pass

    def __call__(self, *args, **kwargs):
        pass

@contextmanager
def _idle_bar():
    try:
        handle = _IdleBarHandle()
        yield handle
    finally:
        pass

try:

    from alive_progress import alive_bar as _bar

    def alive_bar(
        total: Optional[int] = None, *, 
        calibrate: Optional[int] = None, 
        **options: Any
    ):
        if options.get('disable', False):
            return _idle_bar()
        
        return _bar(total, calibrate=calibrate, **options)
except ImportError:

    def alive_bar(
        total: Optional[int] = None, *, 
        calibrate: Optional[int] = None, 
        **options: Any
    ):
        return _idle_bar()