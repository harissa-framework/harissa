import numpy as np
# import argparse as ap
from pathlib import Path

from harissa.core import Dataset
from harissa.plot.plot_datasets import (
    plot_data_distrib,
    compare_marginals,
    plot_data_umap
)

def visualize(args):
    npz_suffix = '.npz'
    if args.ref_dataset_path.suffix == npz_suffix:
        dataset_ref = Dataset.load(args.ref_dataset_path) 
    else:
        dataset_ref = Dataset.load_txt(args.ref_dataset_path)

    if args.sim_dataset_path.suffix == npz_suffix:
        dataset_sim = Dataset.load(args.sim_dataset_path) 
    else:
        dataset_sim = Dataset.load_txt(args.sim_dataset_path)

    t_ref = np.unique(dataset_ref.time_points)
    t_sim = np.unique(dataset_sim.time_points)

    if not (args.distributions or args.pvalues or args.umap):
        args.distributions = True
        args.pvalues = True

    if args.output is not None:
        output = args.output.with_suffix('')
    else: 
        output = Path(args.ref_dataset_path.stem)

    output.mkdir(parents=True, exist_ok=True)

    if args.distributions:
        plot_data_distrib(
            dataset_ref, 
            dataset_sim, 
            output / 'marginals.pdf', 
            t_ref, 
            t_sim
        )
    
    if args.pvalues:
        compare_marginals(
            dataset_ref,
            dataset_sim,
            output / 'comparison.pdf',
            t_ref,
            t_sim
        )

    if args.umap:
        plot_data_umap(
            dataset_ref, 
            dataset_sim, 
            output / 'umap.pdf',
            t_ref,
            t_sim
        )

    print(output)


def add_subcommand(main_subparsers):
    parser = main_subparsers.add_parser('visualize', help='visualize help')
    parser.add_argument(
        'ref_dataset_path',
        type=Path, 
        help='path to the reference dataset'
    )
    parser.add_argument(
        'sim_dataset_path',
        type=Path, 
        help='path to the simulated dataset'
    )
    parser.add_argument(
        '-d','--distributions',
        action='store_true',
        help='plot the marginal distributions of the simulated genes'
    )
    parser.add_argument(
        '-p', '--pvalues',
        action='store_true',
        help='plot the comparison of the marginals '
             'using a Kolmogorov-Smornov test'
    )
    parser.add_argument(
        '-u', '--umap',
        action='store_true',
        help='plot the UMAP reduction of the simulated dataset'
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='output path. It is a directory where pdf files are saved.'
    )

    parser.set_defaults(run=visualize)