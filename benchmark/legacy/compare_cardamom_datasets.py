from pathlib import Path
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from harissa.benchmark.generators import DatasetsGenerator
from harissa.plot.plot_datasets import compare_average_traj
from harissa.plot.plot_network import plot_network, build_pos
from harissa.utils.progress_bar import alive_bar

def compare(path, output, include, exclude):
    
    gen_cached = DatasetsGenerator(path=path)
    gen_cached.networks.include = include
    gen_cached.networks.exclude = exclude

    datasets_per_network = {}
    nb_datasets = 0
    for key in gen_cached:
        if key[0] not in datasets_per_network:
            datasets_per_network[key[0]] = []
        
        datasets_per_network[key[0]].append(key[1])
        nb_datasets += 1
    
    datasets_per_network = { 
        net_name:sorted(datasets_per_network[net_name])
        for net_name in sorted(datasets_per_network)
    }

    gen = DatasetsGenerator(n_datasets=datasets_per_network)
    gen.networks.path = path
    gen.networks.include = include
    gen.networks.exclude = exclude

    network_width = 2
    average_plot_width = 8
    width_ratios = [network_width, average_plot_width, average_plot_width]
    
    row_height = 2
    
    size = (int(np.sum(width_ratios)), row_height * nb_datasets)

    fig = plt.figure(figsize=size, layout='constrained')
    fig.suptitle(
        '''Comparison of average trajectory.
        On the left the loaded dataset and on the right the generated one.'''
    )
    grid = gs.GridSpec(nb_datasets, 3, figure=fig, width_ratios=width_ratios)

    
    with alive_bar(nb_datasets, title='Plotting') as bar:
        i = 0
        for network_name, datasets in datasets_per_network.items():
            for j, dataset_name in enumerate(datasets):
                axs = [fig.add_subplot(grid[i+j, k]) for k in range(3)] 

                ax_network = axs[0]
                ax_pos = ax_network.get_position()
                scale = 0.8 / min(ax_pos.width, ax_pos.height)

                network, dataset_cached = gen_cached[network_name,dataset_name]
                dataset = gen[network_name, dataset_name][1]

                inter = network.interaction

                layout = network.layout 
                if layout is None:
                    layout = build_pos(inter)

                plot_network(inter, layout, axes=ax_network, scale=scale)
                ax_network.set_axis_on()
                ax_network.set_frame_on(False)
                ax_network.set_xticks([])
                ax_network.set_yticks([])
                ax_network.set_ylabel(f'{network_name}-{dataset_name}')

                compare_average_traj(dataset_cached, dataset, axs=axs[1:])
                
                bar()
            
            i += len(datasets)
    
    fig.savefig(output)

def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-p', '--path',
        type=Path
    )
    parser.add_argument(
        '-i', '--include',
        nargs='*',
        default=['*']
    )
    parser.add_argument(
        '-e', '--exclude',
        nargs='*',
        default=[]
    )
    parser.add_argument(
        '-o', '--output',
        type=Path
    )
    args = parser.parse_args()

    if args.path is None:
        path = Path(__file__).parent.parent / 'cardamom_datasets'
    else:
        path = args.path

    if args.output is None:
        output = Path('comparison_dataset.pdf')
    else:
        output = args.output

    compare(path, output, args.include, args.exclude)

main()