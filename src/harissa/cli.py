import argparse as ap

from harissa import __version__
import harissa.infer_cli as infer_cli
import harissa.simulate_cli as simulate_cli

def main():
    parser = ap.ArgumentParser(
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
        help='command help',
        required=True
    )
    infer_cli.add_subcommand(subparsers)
    simulate_cli.add_subcommand(subparsers)

    # parse sys.argv and run the command before exiting
    args = parser.parse_args()
    args.run(args)
    parser.exit()


if __name__ == '__main__':
    main()
