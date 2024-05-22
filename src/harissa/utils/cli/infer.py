import numpy as np
from pathlib import Path
import argparse as ap

from harissa import NetworkModel, NetworkParameter
from harissa.inference import default_inference, Hartree, Cardamom
from harissa.plot import build_pos, plot_network
from harissa.core import Dataset

export_choices = ('npz', 'txt')

def create_hartree(args):
    options = {'verbose' : args.verbose}
    if args.penalization is not None:
        options['penalization_strength'] = args.penalization
    if args.tolerance is not None:
        options['tolerance'] = args.tolerance
    if args.max_iteration is not None:
        options['max_iteration'] = args.max_iteration
    if args.use_numba is not None:
        options['use_numba'] = args.use_numba
    return Hartree(**options)

def create_cardamom(args):
    options = {'verbose' : args.verbose}
    if args.threshold is not None:
        options['threshold'] = args.threshold
    if args.pseudo_l1_coeff is not None:
        options['pseudo_l1_coeff'] = args.pseudo_l1_coeff
    if args.penalization is not None:
        options['penalization'] = args.penalization
    if args.tolerance is not None:
        options['tolerance'] = args.tolerance
    if args.max_iteration is not None:
        options['max_iteration'] = args.max_iteration
    if args.use_numba is not None:
        options['use_numba'] = args.use_numba

    return Cardamom(**options)
        
def infer(args):
    model = NetworkModel(inference=args.create_inference(args))
    npz_suffix = '.npz'
    if args.dataset_path.suffix == npz_suffix:
        dataset = Dataset.load(args.dataset_path) 
    else:
        dataset = Dataset.load_txt(args.dataset_path)

    if args.network_path is not None:
        model.parameter = (
            NetworkParameter.load(args.network_path) 
            if args.network_path.suffix == npz_suffix
            else NetworkParameter.load_txt(args.network_path) 
        )
    
    res = model.fit(dataset)

    if args.output is not None:
        output = args.output.with_suffix('')
    else: 
        output = Path(args.dataset_path.stem + '_inference_result')

    print(
        res.save(output, args.save_extra) if args.format == 'npz' else
        res.save_txt(output, args.save_extra)
    )
    
    if args.save_plot:
        inter = (np.abs(model.interaction) > args.cut_off) * model.interaction
        plot_network(inter, build_pos(inter), file=output.with_suffix('.pdf'))

def add_export_options(parser, plot_option = False):
    parser.add_argument(
        '-f', '--format',
        choices=export_choices,
        default='npz',
        help="output's format."
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        # default=ap.SUPPRESS,
        help='output path. It is a directory if the format is txt'
             ' else it is a .npz file.'
    )
    if plot_option:
        parser.add_argument('--save-plot', action='store_true')    

def add_subcommand(main_subparsers):
    # Infer parser
    parser = main_subparsers.add_parser(
        'infer', 
        help='infer help',
        formatter_class=ap.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('dataset_path', type=Path, help='path to data file')
    parser.add_argument('-n', '--network_path', type=Path, help='path to network file')
    add_export_options(parser, True)
    parser.add_argument(
        '--cut-off',
        type=float,
        default=1.0,
        help='method help'
    )
    parser.add_argument('--save-extra', action='store_true')
    parser.set_defaults(
        create_inference=lambda args: default_inference()
    )
    # set command function (called in the main of cli.py) 
    parser.set_defaults(run=infer)

    # Inference methods hartree (cardamom TODO)
    subparsers = parser.add_subparsers(
        title='inference methods', 
        required= False
    )
    hartree_parser = subparsers.add_parser('hartree')
    cardamom_parser = subparsers.add_parser('cardamom')

    # Hartree parser
    hartree_parser.add_argument('-p', '--penalization', type=float)
    hartree_parser.add_argument('-t', '--tolerance', type=float)
    hartree_parser.add_argument('-n', '--max-iteration', type=int)
    hartree_parser.add_argument('-v', '--verbose', action='store_true')
    hartree_parser.add_argument('--use-numba', action='store_true')
    hartree_parser.add_argument('--no-use-numba',action='store_false',dest='use_numba')
    hartree_parser.set_defaults(create_inference=create_hartree)

    # Cardamom parser
    cardamom_parser.add_argument('--threshold', type=float)
    cardamom_parser.add_argument('-l', '--pseudo-l1-coeff', type=float)
    cardamom_parser.add_argument('-p', '--penalization', type=float)
    cardamom_parser.add_argument('-t', '--tolerance', type=float)
    cardamom_parser.add_argument('-n', '--max-iteration', type=int)
    cardamom_parser.add_argument('-v', '--verbose', action='store_true')
    cardamom_parser.add_argument('--use-numba', action='store_true')
    cardamom_parser.add_argument('--no-use-numba',action='store_false',dest='use_numba')
    cardamom_parser.set_defaults(create_inference=create_cardamom)
