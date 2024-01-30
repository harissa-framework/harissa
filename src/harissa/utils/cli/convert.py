from pathlib import Path
from harissa.utils.npz_io import convert 


def add_subcommand(main_subparsers):
    parser = main_subparsers.add_parser(
        'convert',
        help='convert help'
    )

    parser.add_argument(
        'path', 
        type=Path,
        help='path to convert. It is a .npz file or a directory.'
    )

    parser.set_defaults(run=lambda args: convert(args.path))