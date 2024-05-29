"""
harissa.utils.cli
-----------------

Command line interface for the Harissa package.
"""
import sys
import traceback
import argparse as ap

from harissa import __version__
import harissa.utils.cli.infer as cli_infer
import harissa.utils.cli.trajectory as cli_trajectory
import harissa.utils.cli.dataset as cli_dataset
import harissa.utils.cli.visualize as cli_visualize
import harissa.utils.cli.convert as cli_convert
import harissa.utils.cli.template as cli_template

# Handle exceptions with user-friendly traceback:
# this may be moved later to specific end-user scripts
# ====================================================
def _excepthook(exc_type, exc_value, exc_traceback):
    """Show minimal traceback for exceptions."""
    traceback.print_exception(
        exc_type, 
        value=exc_value,
        tb=exc_traceback, 
        limit=1
    )
sys.excepthook = _excepthook

def main():
    parser = ap.ArgumentParser(
        prog='harissa',
        description='Tools for mechanistic gene network inference '
                    'from single-cell data',
        fromfile_prefix_chars='@'
    )
    parser.convert_arg_line_to_args = lambda arg_line: arg_line.split()

    parser.add_argument(
        '-V', '--version', 
        action='version', 
        version=__version__
    )
    
    # Sub commands infer and simulate
    subparsers = parser.add_subparsers(
        title='commands',
        # help='command help',
        required=True
    )
    cli_infer.add_subcommand(subparsers)
    cli_trajectory.add_subcommand(subparsers)
    cli_dataset.add_subcommand(subparsers)
    cli_visualize.add_subcommand(subparsers)
    cli_convert.add_subcommand(subparsers)
    cli_template.add_subcommand(subparsers)

    # parse sys.argv and run the command before exiting
    args = parser.parse_args()
    args.run(args)
    parser.exit()

if __name__ == '__main__':
    main()
