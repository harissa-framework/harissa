import argparse
from pathlib import Path
import numpy as np

import harissa
import harissa.simulation
import harissa.inference
import harissa.graphics

def infer(args):
    match args.method:
        case 'hartree':
            inf = harissa.inference.Hartree()
        case _:
            inf = harissa.inference.default_inference()
    
    model = harissa.NetworkModel(inference=inf)
    print('inferring ...')
    model.fit(np.loadtxt(args.data_path))

    inter = (np.abs(model.interaction) > args.cutoff) * model.interaction
    
    pos = harissa.graphics.build_pos(inter)
    harissa.graphics.plot_network(inter, pos, scale=2)
    print('done')

def simulate(args):
    print('simulate')
    match args.method:
        case 'bursty':
            sim = harissa.simulation.BurstyPDMP()
        case 'ode':
            sim = harissa.simulation.ApproxODE()
        case _:
            sim = harissa.simulation.default_simulation()
    
    model = harissa.NetworkModel(None, simulation=sim)
    model.simulate()




def main():
    main_parser = argparse.ArgumentParser(
        description='Tools for mechanistic gene network inference '
                    'from single-cell data'
    )

    main_parser.add_argument(
        '-V', '--version', 
        action='version', 
        version=harissa.__version__)
    
    subparsers = main_parser.add_subparsers(
        title='subcommands',
        help='sub-command help',
        required=True
    )

    infer_parser = subparsers.add_parser('infer', help='infer help')
    infer_parser.add_argument('data_path', type=Path, help="path to data file")
    infer_parser.add_argument(
        '-m', '--method', 
        choices=['hartree'],
        help='method help'
    )
    infer_parser.add_argument(
        '--cutoff',
        type=float,
        default=0.0,
        help='method help'
    )
    infer_parser.set_defaults(func=infer)

    simulate_parser = subparsers.add_parser('simulate', help='simulate help')
    # simulate_parser.add_argument(
    #     'filename', 
    #     type=argparse.FileType(),
    #     help="path to data file"
    # )
    simulate_parser.add_argument(
        '-m', '--method', 
        choices=['bursty', 'ode'],
        help='method help'
    )
    simulate_parser.set_defaults(func=simulate)
    
    args = main_parser.parse_args()
    args.func(args)
    main_parser.exit()


if __name__ == '__main__':
    main()
