import shlex
import sys

def cmd_to_args(cmd):
    argv_str = f"import sys;import shlex;sys.argv=shlex.split('{cmd}');"
    main_str = 'from harissa.utils.cli import main;main()'
    return shlex.split(f'{sys.executable} -c "{argv_str}{main_str}"')  