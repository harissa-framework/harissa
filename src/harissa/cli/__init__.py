import argparse as ap

from harissa import __version__
import harissa.cli.infer as cli_infer
import harissa.cli.trajectory as cli_trajectory
import harissa.cli.dataset as cli_dataset
import harissa.cli.convert as cli_convert

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
    cli_convert.add_subcommand(subparsers)

    # parse sys.argv and run the command before exiting
    args = parser.parse_args()
    args.run(args)
    parser.exit()