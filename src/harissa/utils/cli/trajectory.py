import numpy as np
from pathlib import Path
import argparse as ap

from harissa import NetworkModel, NetworkParameter
from harissa.simulation import default_simulation, BurstyPDMP, ApproxODE
from harissa.plot import plot_simulation
from harissa.utils.npz_io import ParamInfos, load_dir, load_npz
from harissa.utils.cli.infer import add_export_options, export_choices

simulation_param_names = {
    'time_points': ParamInfos(True, np.float_, 1),
    'initial_state': ParamInfos(False, np.float_, 2),
    'initial_time': ParamInfos(False, np.float_, 0)
}

def _create_load_simulation_parameter(load_fn):
    def load_simulation_parameter(path: str | Path) -> dict:
        sim_param = load_fn(path, simulation_param_names)
        sim_param['time_points'] = np.unique(sim_param['time_points'])

        return sim_param
    
    return load_simulation_parameter

load_simulation_parameter_txt = _create_load_simulation_parameter(load_dir)
load_simulation_parameter = _create_load_simulation_parameter(load_npz)

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

    if args.network_parameter_path.is_dir():
        network_param = NetworkParameter.load_txt(args.network_parameter_path)
    else:
        network_param = NetworkParameter.load(args.network_parameter_path)

    if args.simulation_parameter_path.is_dir():
        param = load_simulation_parameter_txt(args.simulation_parameter_path)
    else:
        param = load_simulation_parameter(args.simulation_parameter_path)

    model= NetworkModel(network_param, simulation=args.create_simulation(args))

    if args.burn_in is not None:
        # override initial state
        param['initial_state'] = model.burn_in(args.burn_in)
    
    res = model.simulate(**param)

    print(
        res.save_txt(output) if args.format == export_choices[1] else
        res.save(output)
    )

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
        help='burn in duration. (override the initial state)'
    )
    add_export_options(parser, True)
    # set command function (called in the main of cli.py) 
    parser.set_defaults(run=simulate)
    
    add_methods(parser) 