import numpy as np
from pathlib import Path
import argparse as ap

from harissa import NetworkModel
from harissa.inference import default_inference, Hartree
from harissa.graphics import build_pos, plot_network
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

# TODO
def create_cardamom(args):
    ...


def save_extra_hartree(output, res, args):
    if not args.save_extra:
        return
    
    basal_time = {str(t):val for t, val in res.basal_time.items()}
    inter_time = {str(t):val for t, val in res.interaction_time.items()}

    if args.format == export_choices[0]:
        output = str(output) + '_extra'
        np.savez_compressed(output + '_basal_time', **basal_time)
        np.savez_compressed(output + '_interaction_time', **inter_time)
        np.savez_compressed(output + '_y', y=res.y)
    else:
        output = output / 'extra'
        (output / 'basal_time').mkdir(parents=True, exist_ok=True)
        (output / 'interaction_time').mkdir(exist_ok=True)

        suffix = '.txt'
        for time, value in basal_time.items():
            np.savetxt(
                (output / 'basal_time' / f't_{time}').with_suffix(suffix), 
                value
            )
        for time, value in inter_time.items():
            np.savetxt(
                (output/'interaction_time'/f't_{time}').with_suffix(suffix), 
                value
            )
        np.savetxt(output / 'y', res.y)
        
def infer(args):
    model = NetworkModel(inference=args.create_inference(args))
    if args.dataset_path.suffix == '.npz':
        dataset = Dataset.load(args.dataset_path) 
    else:
        dataset = Dataset.load_txt(args.dataset_path) 
    
    res = model.fit(dataset)

    if args.output is not None:
        output = args.output.with_suffix('')
    else: 
        output = Path(args.dataset_path.stem + '_inference_result')

    print(
        res.parameter.save(output) if args.format == 'npz' else
        res.parameter.save_txt(output)
    )
    args.save_extra_info(output, res, args)

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

    parser.add_argument('dataset_path', type=Path, help="path to data file")
    add_export_options(parser, True)
    parser.add_argument(
        '--cut-off',
        type=float,
        default=1.0,
        help='method help'
    )
    parser.set_defaults(
        create_inference=lambda args: default_inference()
    )
    parser.set_defaults(save_extra_info=lambda res, output, format: None)
    # set command function (called in the main of cli.py) 
    parser.set_defaults(run=infer)

    # Inference methods hartree (cardamom TODO)
    subparsers = parser.add_subparsers(
        title='inference methods', 
        required= False
    )
    hartree_parser = subparsers.add_parser('hartree')

    # Hartree parser
    hartree_parser.add_argument('-p', '--penalization', type=float)
    hartree_parser.add_argument('-t', '--tolerance', type=float)
    hartree_parser.add_argument('-n', '--max-iteration', type=int)
    hartree_parser.add_argument('-v', '--verbose', action='store_true')
    hartree_parser.add_argument('--use-numba', action=ap.BooleanOptionalAction)
    hartree_parser.add_argument('--save-extra', action='store_true')
    hartree_parser.set_defaults(create_inference=create_hartree)
    hartree_parser.set_defaults(save_extra_info=save_extra_hartree)