import numpy as np
import argparse as ap
from pathlib import Path

from harissa import NetworkModel, NetworkParameter
from harissa.core.dataset import Dataset
from harissa.processing import binarize
from harissa.utils.progress_bar import alive_bar
from harissa.utils.cli.infer import add_export_options, export_choices
from harissa.utils.cli.trajectory import add_methods

def simulate_dataset(args):
    if args.output is not None:
        output = args.output.with_suffix('')
    else: 
        output = Path(args.dataset_path.stem + '_dataset_result')

    if args.network_parameter_path.is_dir():
        network_param = NetworkParameter.load_txt(args.network_parameter_path)
    else:
        network_param = NetworkParameter.load(args.network_parameter_path)

    if args.dataset_path.suffix == '.npz':
        dataset = Dataset.load(args.dataset_path)        
    else:
        dataset = Dataset.load_txt(args.dataset_path)

    model= NetworkModel(network_param, simulation=args.create_simulation(args))

    data_prot = binarize(dataset).count_matrix
    data_sim = np.empty(dataset.count_matrix.shape, dtype=np.uint)

    # copy 1rst column because time points will be replaced by stimuli 
    non_zero_time_points = dataset.time_points != 0.0

    # set stimuli in 1rst column with non zero time points
    # dataset.count_matrix[:, 0] = non_zero_time_points
    data_prot[:, 0] = non_zero_time_points
    data_sim[:, 0] = non_zero_time_points

    # extract cell indices at time_points == 0
    cell_indices_at_t0 = np.flatnonzero(~non_zero_time_points)

    with alive_bar(dataset.time_points.size, title='Processing cells') as bar:
        for cell_index in range(dataset.time_points.size):
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
                        initial_state=np.vstack((M0, P0), dtype=np.float64)
                    ).rna_levels[0, 1:]
                )
            bar()

            
    dataset.count_matrix[:] = data_sim
    print(
        dataset.save_txt(output) if args.format == export_choices[1] else 
        dataset.save(output)
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
