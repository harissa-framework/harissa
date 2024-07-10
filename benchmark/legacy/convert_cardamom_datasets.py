from pathlib import Path
import numpy as np
import argparse as ap
from shutil import make_archive
from tempfile import TemporaryDirectory

from harissa import Dataset
from harissa.utils.progress_bar import alive_bar
from harissa.benchmark.generators.networks import NetworksGenerator, tree
from harissa.plot import build_pos

def convert(cardamom_article, output_dir):
    deterministic_networks = ['BN8', 'CN5', 'FN4', 'FN8']
    network_gen = NetworksGenerator(
        include=deterministic_networks, 
        verbose=True
    )
    network_gen.save(output_dir)

    with alive_bar(len(list(cardamom_article.iterdir()))) as bar:
        for folder in cardamom_article.iterdir():
            folder_name = folder.name
            old_datasets = [
                p for p in (folder / 'Data').iterdir() if p.suffix == '.txt'
            ]
            datasets_output = output_dir / 'datasets' / folder_name
            
            if folder_name in deterministic_networks:
                # save datasets
                for i, path in enumerate(old_datasets, 1):
                    old_dataset = np.loadtxt(path, dtype=int, delimiter='\t')
                    dataset = Dataset(
                        old_dataset[0, 1:].astype(np.float_),
                        old_dataset[1:, 1:].T.astype(np.uint)
                    )
                    
                    datasets_output.mkdir(parents=True, exist_ok=True)
                    dataset.save(datasets_output / f'd{i}.npz')
            else:
                assert folder_name.startswith('Trees')

                inters = [
                    p for p in (folder / 'True').iterdir() if p.suffix=='.npy'
                ]
                networks_output = output_dir / 'networks' / folder_name
                # save networks interaction
                for path in inters:
                    tree_name = f'{path.stem.split("_")[1]}.npz'
                    inter = np.load(path) * 10.0
                    network = tree(inter.shape[1] - 1)
                    # override interaction matrix
                    network.interaction[:] = inter
                    network.layout = build_pos(inter)
                    network.save(networks_output / tree_name)
                # save datasets
                for path in old_datasets:
                    tree_name = path.stem.split("_")[1]
                    old_dataset = np.loadtxt(path, dtype=int, delimiter='\t')
                    dataset = Dataset(
                        old_dataset[0, 1:].astype(np.float_),
                        old_dataset[1:, 1:].T.astype(np.uint)
                    )
                    output = datasets_output/ tree_name / 'd1.npz'
                    output.parent.mkdir(parents=True, exist_ok=True)
                    dataset.save(output)
            bar()

def archive(cardamom_article, output_dir, archive_format):
    with TemporaryDirectory() as tmp_dir:
        convert(cardamom_article, Path(tmp_dir))
        with alive_bar(title='Archiving', monitor=False, stats=False) as bar:
            make_archive(str(output_dir), archive_format, tmp_dir)
            bar()

def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-o', '--output',
        type=Path
    )
    parser.add_argument(
        '-f', '--format',
        choices=('zip', 'tar', 'gztar'),
    )
    args = parser.parse_args()

    cur_file_dir = Path(__file__).parent
    cardamom_article = cur_file_dir / 'results_cardamom_article'
    output_dir = (
        cur_file_dir.parent / 'cardamom_datasets' 
        if args.output is None else args.output
    )

    if args.format is None:
        convert(cardamom_article, output_dir)
    else:
        archive(cardamom_article, output_dir, args.format)

main()