from pathlib import Path
import argparse as ap

from harissa import NetworkModel
from harissa.simulation import default_simulation, BurstyPDMP, ApproxODE
from harissa.graphics import plot_simulation
from harissa.utils.npz_io import (load_simulation_parameter, 
                                  load_network_parameter,
                                  save_simulation_result, export_format)

def create_bursty(args):
    options = {'verbose' : args.verbose, 'use_numba': args.use_numba}
    if args.thin_adapt is not None:
        options['thin_adapt'] = args.thin_adapt
    return BurstyPDMP(**options)

def create_ode(args):
    return ApproxODE(verbose=args.verbose, use_numba=args.use_numba)

def simulate(args):
    if args.output is not None:
        output = args.output.with_suffix('')
    else: 
        output = Path(
            args.simulation_parameter_path.stem + '_simulation_result'
        )

    model= NetworkModel(
        load_network_parameter(args.network_parameter_path),
        simulation=args.create_simulation(args)
    )
    
    res = model.simulate(
        **load_simulation_parameter(
            args.simulation_parameter_path, 
            args.burn_in
        )
    )
    print(save_simulation_result(output, res, args.format))

    if args.save_plot:
        fig = plot_simulation(res)
        fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')

def add_methods(parser):
    parser.set_defaults(create_simulation=lambda args: default_simulation())

    # Simulation methods bursty and ode
    subparsers = parser.add_subparsers(
        title='simulation methods (optional)', 
        required=False,
        help='specify it to choose the simulation method '
             'and to parametrize it. ' 
             'If not specified the bursty method is used by default.'
    )
    bursty_parser = subparsers.add_parser('bursty')
    ode_parser = subparsers.add_parser('ode')

    # Bursty parser
    bursty_parser.add_argument('--thin-adapt', action=ap.BooleanOptionalAction)
    bursty_parser.add_argument('-v', '--verbose', action='store_true')
    bursty_parser.add_argument('--use-numba', action='store_true')
    bursty_parser.set_defaults(create_simulation=create_bursty)
    
    # Ode parser
    ode_parser.add_argument('-v', '--verbose', action='store_true')
    ode_parser.add_argument('--use-numba', action='store_true')
    ode_parser.set_defaults(create_simulation=create_ode)

def add_subcommand(main_subparsers):
    # Simulate parser
    parser = main_subparsers.add_parser(
        'trajectory',
        help='simulate a trajectory',
        formatter_class=ap.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'simulation_parameter_path', 
        type=Path,
        help='path to simulation parameter. It is a .npz file or a directory.'
    )
    parser.add_argument(
        'network_parameter_path', 
        type=Path,
        help='path to network parameter. It is a .npz file or a directory.'
    )
    parser.add_argument(
        '-b', '--burn-in',
        type=float,
        # default=ap.SUPPRESS,
        help='burn in parameter.'
    )
    parser.add_argument(
        '-f', '--format',
        choices=export_format,
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
    parser.add_argument('--save-plot', action='store_true')
    # set command function (called in the main of cli.py) 
    parser.set_defaults(run=simulate)
    
    add_methods(parser) 