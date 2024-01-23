import numpy as np
from pathlib import Path
import argparse as ap

from harissa import NetworkModel
from harissa.inference import default_inference, Hartree

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

def save(output_path, output_dict, output_format):
    if output_format == 'txt' or output_format == 'txt_col':
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == 'npz':
        np.savez(output_path, **output_dict)
    elif output_format == 'npz_compressed':
        np.savez_compressed(output_path, **output_dict)
    else:
        for key, value in output_dict.items():
            np.savetxt(
                (output_path / key).with_suffix('.txt'), 
                value if output_format == 'txt' else np.atleast_2d(value)
            )


def save_extra_hartree(res, output, args):
    if not args.save_extra:
        return
    
    basal_time = {str(t):val for t, val in res.basal_time.items()}
    inter_time = {str(t):val for t, val in res.interaction_time.items()}

    is_txt_format = args.format == 'txt' 
    if is_txt_format or args.format == 'txt_col':
        output = output / 'extra'
        (output / 'basal_time').mkdir(parents=True, exist_ok=True)
        (output / 'interaction_time').mkdir(exist_ok=True)
    else:
        output = str(output) + '_extra'

    if args.format == 'npz':
        np.savez(output + '_basal_time', **basal_time)
        np.savez(output + '_interaction_time', **inter_time)
        np.savez(output + '_y', y=res.y)
    elif args.format == 'npz_compressed':
        np.savez_compressed(output + '_basal_time', **basal_time)
        np.savez_compressed(output + '_interaction_time', **inter_time)
        np.savez_compressed(output + '_y', y=res.y)
    else:
        for time, value in basal_time.items():
            np.savetxt(
                (output / 'basal_time' / f't_{time}').with_suffix('.txt'), 
                value if is_txt_format else np.atleast_2d(value)
            )
        for time, value in inter_time.items():
            np.savetxt(
                (output/'interaction_time'/f't_{time}').with_suffix('.txt'), 
                value if is_txt_format else np.atleast_2d(value)
            )
        np.savetxt(output / 'y', res.y)
    
def infer(args):
    model = NetworkModel(inference=args.create_inference(args))
    print('inferring ...')
    res = model.fit(np.loadtxt(args.data_path))

    if args.output is not None:
        output = args.output.with_suffix('')
    else: 
        output = Path(args.data_path.stem + '_inference_result')

    save(
        output, 
        {
            'burst_frequency_min': res.parameter.burst_frequency_min,
            'burst_frequency_max': res.parameter.burst_frequency_max,
            'burst_size_inv': res.parameter.burst_size_inv,
            'creation_rna': res.parameter.creation_rna,
            'creation_protein': res.parameter.creation_protein,
            'degradation_rna': res.parameter.degradation_rna,
            'degradation_protein': res.parameter.degradation_protein,
            'basal': res.parameter.basal,
            'interaction': res.parameter.interaction 
        }, 
        args.format
    )

    args.save_extra_info(res, output, args)

    # inter = (np.abs(model.interaction) > args.cut_off) * model.interaction
    
    # pos = harissa.graphics.build_pos(inter)
    # harissa.graphics.plot_network(inter, pos, scale=2)
    print('done')

def add_subcommand(subparsers):
    # Infer parser
    infer_parser = subparsers.add_parser(
        'infer', 
        help='infer help',
        formatter_class=ap.ArgumentDefaultsHelpFormatter
    )

    infer_parser.add_argument('data_path', type=Path, help="path to data file")
    # infer_parser.add_argument(
    #     '--cut-off',
    #     type=float,
    #     default=0.0,
    #     help='method help'
    # )
    infer_parser.add_argument(
        '-f', '--format',
        choices=['npz', 'npz_compressed', 'txt', 'txt_col'],
        default='npz',
        help="output's format."
    )
    infer_parser.add_argument(
        '-o', '--output',
        type=Path,
        default=ap.SUPPRESS,
        help='output path. It is a directory if the format is txt'
             ' else it is a .npz file.'
    )
    infer_parser.set_defaults(
        create_inference=lambda args: default_inference()
    )
    infer_parser.set_defaults(save_extra_info=lambda res, output, format: None)
    # set command function (called in the main of cli.py) 
    infer_parser.set_defaults(run=infer)

    # Inference methods hartree (cardamom TODO)
    infer_subparsers = infer_parser.add_subparsers(
        title='inference methods', 
        required= False
    )
    hartree_parser = infer_subparsers.add_parser('hartree')

    # Hartree parser
    hartree_parser.add_argument('-p', '--penalization', type=float)
    hartree_parser.add_argument('-t', '--tolerance', type=float)
    hartree_parser.add_argument('-n', '--max-iteration', type=int)
    hartree_parser.add_argument('-v', '--verbose', action='store_true')
    hartree_parser.add_argument('--use-numba', action=ap.BooleanOptionalAction)
    hartree_parser.add_argument('--save-extra', action='store_true')
    hartree_parser.set_defaults(create_inference=create_hartree)
    hartree_parser.set_defaults(save_extra_info=save_extra_hartree)