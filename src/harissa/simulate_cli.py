import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse as ap

from harissa import NetworkParameter, NetworkModel
from harissa.simulation import default_simulation, BurstyPDMP, ApproxODE
from harissa.infer_cli import save
from harissa.graphics import plot_simulation

def create_bursty(args):
    options = {'verbose' : args.verbose, 'use_numba': args.use_numba}
    if args.thin_adapt is not None:
        options['thin_adapt'] = args.thin_adapt
    return BurstyPDMP(**options)

def create_ode(args):
    return ApproxODE(verbose=args.verbose, use_numba=args.use_numba)

def load(path, param_names):
    data = None
    if path.is_dir():
        data = {}
        for name, required in param_names.items():
            file_name = (path / name).with_suffix('.txt')
            if required or file_name.exists():
                data[name] = np.loadtxt(file_name)
    elif path.suffix == '.npz':
        with np.load(path) as npz_file:
            data = dict(npz_file)
    else:
        raise RuntimeError(f'{path} must be a .npz file or a directory.')

    return data

def simulate(args):
    network_param_names = {
        'burst_frequency_min': True,
        'burst_frequency_max': True,
        'burst_size_inv': True,
        'creation_rna': True,
        'creation_protein': True,
        'degradation_rna': True,
        'degradation_protein': True,
        'basal': True,
        'interaction': True
    }

    sim_param_names = {
        'time_points': True,
        'M0': False,
        'P0': False
    }

    if args.output is not None:
        output = args.output.with_suffix('')
    else: 
        output = Path(
            args.simulation_parameter_path.stem + '_simulation_result'
        )

    # Preparing model
    data = load(args.network_parameter_path, network_param_names)
    network_param = NetworkParameter(data['basal'].size - 1)

    for key, value in data.items():
        getattr(network_param, key)[:] = value[:]

    model= NetworkModel(network_param, simulation=args.create_simulation(args))
    
    data = load(args.simulation_parameter_path, sim_param_names)
    if args.burn_in is not None:
        data['burn_in'] = args.burn_in
    
    print('simulating...')
    res = model.simulate(**data)
    save(
        output, 
        {
            'time_points': res.time_points,
            'rna_levels': res.rna_levels,
            'protein_levels': res.protein_levels
        },
        args.format
    )

    if args.save_plot:
        plot_simulation(res)
        plt.gcf().savefig(output.with_suffix('.pdf'), bbox_inches='tight')

    print('done')

def add_subcommand(subparsers):
    # Simulate parser
    simulate_parser = subparsers.add_parser(
        'simulate',
        help='simulate help',
        formatter_class=ap.ArgumentDefaultsHelpFormatter
    )
    
    simulate_parser.add_argument(
        'network_parameter_path', 
        type=Path,
        help='path to network parameter. It is a .npz file or a directory.'
    )
    simulate_parser.add_argument(
        'simulation_parameter_path', 
        type=Path,
        help='path to simulation parameter. It is a .npz file or a directory.'
    )
    simulate_parser.add_argument(
        '-b', '--burn-in',
        type=float,
        # default=ap.SUPPRESS,
        help='burn in parameter.'
    )
    simulate_parser.add_argument(
        '-f', '--format',
        choices=['npz', 'npz_compressed', 'txt', 'txt_col'],
        default='npz',
        help="output's format."
    )
    simulate_parser.add_argument(
        '-o', '--output',
        type=Path,
        # default=ap.SUPPRESS,
        help='output path. It is a directory if the format is txt'
             ' else it is a .npz file.'
    )
    simulate_parser.add_argument('--save-plot', action='store_true')
    simulate_parser.set_defaults(
        create_simulation=lambda args: default_simulation()
    )
    # set command function (called in the main of cli.py) 
    simulate_parser.set_defaults(run=simulate)
    
    # Simulation methods bursty and ode
    simulate_subparsers = simulate_parser.add_subparsers(
        title='simulation methods (optional)', 
        required=False,
        help='specify it to choose the simulation method '
             'and to parametrize it. ' 
             'If not specified the bursty method is used by default.'
    )
    bursty_parser = simulate_subparsers.add_parser('bursty')
    ode_parser = simulate_subparsers.add_parser('ode')

    # Bursty parser
    bursty_parser.add_argument('--thin-adapt', action=ap.BooleanOptionalAction)
    bursty_parser.add_argument('-v', '--verbose', action='store_true')
    bursty_parser.add_argument('--use-numba', action='store_true')
    bursty_parser.set_defaults(create_simulation=create_bursty)
    
    # Ode parser
    ode_parser.add_argument('-v', '--verbose', action='store_true')
    ode_parser.add_argument('--use-numba', action='store_true')
    ode_parser.set_defaults(create_simulation=create_ode) 