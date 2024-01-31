import numpy as np
import argparse as ap
from pathlib import Path
from alive_progress import alive_it

from harissa import NetworkModel
from harissa.utils.npz_io import (load_network_parameter, load_dataset, 
                                  save, export_format)
from harissa.utils.processing import binarize
from harissa.utils.cli.trajectory import add_methods

def simulate_dataset(args):
    if args.output is not None:
        output = args.output.with_suffix('')
    else: 
        output = Path(args.data_path.stem + '_dataset_result')

    model = NetworkModel(
        load_network_parameter(args.network_parameter_path),
        simulation=args.create_simulation(args)
    )

    dataset = load_dataset(args.data_path)
    data_prot = binarize(dataset).count_matrix
    data_sim = np.empty(dataset.count_matrix.shape, dtype=np.uint)

    # copy 1rst column because time points will be replaced by stimuli 
    non_zero_time_points = dataset.time_points != 0.0

    # set stimuli in 1rst column with non zero time points
    dataset.count_matrix[:, 0] = non_zero_time_points
    data_prot[:, 0] = non_zero_time_points
    data_sim[:, 0] = non_zero_time_points

    # extract cell indices at time_points == 0
    cell_indices_at_t0 = np.flatnonzero(~non_zero_time_points)

    for cell_index in alive_it(range(dataset.time_points.size)):
        cell_time = dataset.time_points[cell_index]
        if cell_time == 0.0:
            data_sim[cell_index, 1:] = dataset.count_matrix[cell_index, 1:]
        else:
            cell_index_at_t0 = np.random.choice(cell_indices_at_t0)
            data_sim[cell_index, 1:] = np.random.poisson(
                model.simulate(
                    cell_time, 
                    M0=dataset.count_matrix[cell_index_at_t0], 
                    P0=data_prot[cell_index_at_t0]
                ).rna_levels[0]
            )

    print(
        save(
            output, 
            {'time_points': dataset.time_points, 'count_matrix': data_sim},
            args.format
        )
    )
    



def add_subcommand(main_subparsers):
    parser = main_subparsers.add_parser(
        'dataset', 
        help='simulate a dataset',
        formatter_class=ap.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('data_path', type=Path, help="path to data file")
    parser.add_argument(
        'network_parameter_path', 
        type=Path,
        help='path to network parameter. It is a .npz file or a directory.'
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

    parser.set_defaults(run=simulate_dataset)

    add_methods(parser)
