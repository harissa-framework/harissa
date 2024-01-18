import argparse as ap
from pathlib import Path
import numpy as np

import harissa
import harissa.simulation
import harissa.inference
# import harissa.graphics

# Inferences creation
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
    return harissa.inference.Hartree(**options)

def save_extra_hartree(res, output, args):
    if not args.save_extra:
        return
    
    basal_time = {str(t):val for t, val in res.basal_time.items()}
    inter_time = {str(t):val for t, val in res.interaction_time.items()}

    is_txt_format = args.format == 'txt' 
    if is_txt_format or args.format == 'txt_c':
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
    

# TODO
def create_cardamom(args):
    ...

# Simulations creation 
def create_bursty(args):
    options = {'verbose' : args.verbose, 'use_numba': args.use_numba}
    if args.thin_adapt is not None:
        options['thin_adapt'] = args.thin_adapt
    return harissa.simulation.BurstyPDMP(**options)

def create_ode(args):
    return harissa.simulation.ApproxODE(
        verbose=args.verbose, 
        use_numba=args.use_numba
    )

# infer command
def infer(args):
    model = harissa.NetworkModel(inference=args.create_inference(args))
    print('inferring ...')
    res = model.fit(np.loadtxt(args.data_path))
    param_dict = {
        'burst_frequency_min': res.parameter.burst_frequency_min,
        'burst_frequency_max': res.parameter.burst_frequency_max,
        'burst_size_inv': res.parameter.burst_size_inv,
        'creation_rna': res.parameter.creation_rna,
        'creation_protein': res.parameter.creation_protein,
        'degradation_rna': res.parameter.degradation_rna,
        'degradation_protein': res.parameter.degradation_protein,
        'basal': res.parameter.basal,
        'interaction': res.parameter.interaction 
    }

    if args.output is not None:
        if args.format == 'txt' or args.format == 'txt_c':
            output = args.output
            args.output.mkdir(parents=True, exist_ok=True)
        else:
            output = args.output.with_suffix('')
            args.output.parent.mkdir(parents=True, exist_ok=True)
    else: 
        output = args.data_path.stem + '_inference_result'

    if args.format == 'npz':
        np.savez(output, **param_dict)
    elif args.format == 'npz_compressed':
        np.savez_compressed(output, **param_dict)
    else:
        for key, value in param_dict.items():
            np.savetxt(
                (output / key).with_suffix('.txt'), 
                value if args.format == 'txt' else np.atleast_2d(value)
            )

    args.save_extra_info(res, output, args)

    # inter = (np.abs(model.interaction) > args.cut_off) * model.interaction
    
    # pos = harissa.graphics.build_pos(inter)
    # harissa.graphics.plot_network(inter, pos, scale=2)
    print('done')

# simulate command
def simulate(args):
    print('simulate')
    return
    model = harissa.NetworkModel(None, simulation=args.create_simulation(args))
    model.simulate()


def main():
    # Main parser
    main_parser = ap.ArgumentParser(
        description='Tools for mechanistic gene network inference '
                    'from single-cell data'
    )

    main_parser.add_argument(
        '-V', '--version', 
        action='version', 
        version=harissa.__version__)
    
    # Sub commands infer and simulate
    subparsers = main_parser.add_subparsers(
        title='commands',
        help='command help',
        required=True
    )
    infer_parser = subparsers.add_parser('infer', help='infer help')
    simulate_parser = subparsers.add_parser('simulate', help='simulate help')
    
    # Infer parser
    infer_parser.add_argument('data_path', type=Path, help="path to data file")
    # infer_parser.add_argument(
    #     '--cut-off',
    #     type=float,
    #     default=0.0,
    #     help='method help'
    # )
    infer_parser.add_argument(
        '-o', '--output',
        type=Path,
        help='output directory or file. It is a directory if the format is txt'
             ' else it is a file.'
    )
    infer_parser.add_argument(
        '-f', '--format',
        choices=['npz', 'npz_compressed', 'txt', 'txt_c'],
        default='npz',
        help="output's format. Default format is npz."
    )
    infer_parser.set_defaults(
        create_inference=lambda args: harissa.inference.default_inference()
    )
    infer_parser.set_defaults(save_extra_info=lambda res, output, format: None)
    infer_parser.set_defaults(run=infer)
    
    # Inference methods hartree (cardamom TODO)
    infer_subparser = infer_parser.add_subparsers(
        title='inference methods', 
        required= False
    )
    hartree_parser = infer_subparser.add_parser('hartree')

    # Hartree parser
    hartree_parser.add_argument('-p', '--penalization', type=float)
    hartree_parser.add_argument('-t', '--tolerance', type=float)
    hartree_parser.add_argument('-n', '--max-iteration', type=int)
    hartree_parser.add_argument('-v', '--verbose', action='store_true')
    hartree_parser.add_argument('--use-numba', action=ap.BooleanOptionalAction)
    hartree_parser.add_argument('--save-extra', action='store_true')
    hartree_parser.set_defaults(create_inference=create_hartree)
    hartree_parser.set_defaults(save_extra_info=save_extra_hartree)

    # Simulate parser
    # simulate_parser.add_argument(
    #     'filename', 
    #     type=argparse.FileType(),
    #     help="path to data file"
    # )
    simulate_parser.set_defaults(
        create_simulation=lambda args: harissa.simulation.default_simulation()
    )
    simulate_parser.set_defaults(run=simulate)
    
    # Simulation methods bursty and ode
    simulate_subparser = simulate_parser.add_subparsers(
        title='simulation methods', 
        required=False
    )
    bursty_parser = simulate_subparser.add_parser('bursty')
    ode_parser = simulate_subparser.add_parser('ode')

    # Bursty parser
    bursty_parser.add_argument('--thin-adapt', action=ap.BooleanOptionalAction)
    bursty_parser.add_argument('-v', '--verbose', action='store_true')
    bursty_parser.add_argument('--use-numba', action='store_true')
    bursty_parser.set_defaults(create_simulation=create_bursty)
    
    # Ode parser
    ode_parser.add_argument('-v', '--verbose', action='store_true')
    ode_parser.add_argument('--use-numba', action='store_true')
    ode_parser.set_defaults(create_simulation=create_ode)
    
    # parse sys.argv and run the command before exiting
    args = main_parser.parse_args()
    args.run(args)
    main_parser.exit()


if __name__ == '__main__':
    main()
