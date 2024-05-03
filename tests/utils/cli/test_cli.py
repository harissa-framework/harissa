from harissa import __version__
import subprocess

from . import cmd_to_args

def test_help():
    process = subprocess.run(cmd_to_args('harissa -h'))

    assert process.returncode == 0

def test_version():
    process = subprocess.run(
        cmd_to_args('harissa -V'), 
        capture_output=True, 
        text=True
    )

    assert process.returncode == 0
    assert process.stdout.strip() == __version__