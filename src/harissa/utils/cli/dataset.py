import numpy as np
import argparse as ap
from pathlib import Path
from alive_progress import alive_it
from alive_progress.animations.spinners import bouncing_spinner_factory as bsp

from harissa import NetworkModel
from harissa.core.dataset import Dataset
from harissa.utils.npz_io import (load_network_parameter_txt,
                                  load_network_parameter,
                                  load_dataset_txt, 
                                  load_dataset,
                                  save_dataset_txt, 
                                  save_dataset,
                                  suffixes)
from harissa.processing import binarize
from harissa.utils.cli.infer import add_export_options, export_formats
from harissa.utils.cli.trajectory import add_methods

def simulate_dataset(args):
    if args.output is not None:
        output = args.output.with_suffix('')
    else: 
        output = Path(args.dataset_path.stem + '_dataset_result')

    if args.network_parameter_path.is_dir():
        load_network_fn = load_network_parameter_txt
    else:
        load_network_fn = load_network_parameter

    if args.dataset_path.suffix == suffixes[0]:
        load_dataset_fn = load_dataset        
    else:
        load_dataset_fn = load_dataset_txt


    model = NetworkModel(
        load_network_fn(args.network_parameter_path),
        simulation=args.create_simulation(args)
    )

    dataset = load_dataset_fn(args.dataset_path)
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
    
    for cell_index in alive_it(range(dataset.time_points.size), 
                               title='Processing cells',
                               spinner=bsp('🌶', 6, hide=False),
                               receipt=False):
        cell_time = dataset.time_points[cell_index]
        if cell_time == 0.0:
            data_sim[cell_index, 1:] = dataset.count_matrix[cell_index, 1:]
        else:
            cell_index_at_t0 = np.random.choice(cell_indices_at_t0)
            M0 = dataset.count_matrix[cell_index_at_t0]
            P0 = data_prot[cell_index_at_t0]
            data_sim[cell_index, 1:] = np.random.poisson(
                model.simulate(
                    cell_time,
                    initial_state=np.vstack((M0, P0), dtype=np.float_)
                ).rna_levels[0, 1:]
            )

    if args.format == export_formats[1]:
        save_dataset_fn = save_dataset_txt
    else:
        save_dataset_fn = save_dataset 

    print(
        save_dataset_fn(
            output, 
            Dataset(dataset.time_points, data_sim)
        )
    )

def add_subcommand(main_subparsers):
    parser = main_subparsers.add_parser(
        'dataset', 
        help='simulate a dataset',
        formatter_class=ap.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('dataset_path', type=Path, help="path to dataset file")
    parser.add_argument(
        'network_parameter_path', 
        type=Path,
        help='path to network parameter. It is a .npz file or a directory.'
    )
    add_export_options(parser)

    parser.set_defaults(run=simulate_dataset)

    add_methods(parser)
