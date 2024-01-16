import argparse as ap
from pathlib import Path
import numpy as np

import harissa
import harissa.simulation
import harissa.inference
import harissa.graphics

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
    model.fit(np.loadtxt(args.data_path))

    inter = (np.abs(model.interaction) > args.cut_off) * model.interaction
    
    pos = harissa.graphics.build_pos(inter)
    harissa.graphics.plot_network(inter, pos, scale=2)
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
    infer_parser.add_argument(
        '--cut-off',
        type=float,
        default=0.0,
        help='method help'
    )
    infer_parser.set_defaults(
        create_inference=lambda args: harissa.inference.default_inference()
    )
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
    hartree_parser.set_defaults(create_inference=create_hartree)

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
