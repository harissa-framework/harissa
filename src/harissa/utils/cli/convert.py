from pathlib import Path
from harissa.core import Dataset, NetworkParameter, Simulation
from harissa.utils.npz_io import (load_dir,
                                  load_npz,
                                  save_dir,
                                  save_npz)
from harissa.utils.cli.trajectory import simulation_param_names

names_allowed = {
    'dataset': Dataset.param_names,
    'network parameter': NetworkParameter.param_names,
    'simulation parameter': simulation_param_names,
    'simulation result': Simulation.Result.param_names
}

def convert(args) -> None:
    path = args.path
    if not path.exists():
        raise RuntimeError(f"{path} doesn't exist.")

    is_dir = path.is_dir()
    suffixes = ('.npz', '.txt', '.h5ad', '.json')
    if path.suffix == suffixes[1]:
        data = Dataset.load_txt(path).as_dict()
    elif path.suffix == suffixes[2]:
        data = Dataset.load_h5ad(path).as_dict()
    elif path.suffix == suffixes[3]:
        data = NetworkParameter.load_json(path).as_dict()
    elif is_dir or path.suffix == suffixes[0]:
        load_fn = load_dir if is_dir else load_npz
        loads = []
        for key, param_names in names_allowed.items():
            try:
                loads.append((key, load_fn(path, param_names)))
            except RuntimeError:
                pass
        if len(loads) == 0:
            raise RuntimeError(f'Try to convert something that is not a '
                               f'{"nor a ".join(names_allowed.keys())}.')
        elif len(loads) > 1:
            raise RuntimeError('Try to convert a '
                               f'{"and a ".join(map(lambda t: t[0], loads))}.'
                               'Chose one.')
        else:
            data = loads[0][1]
    else:
        raise RuntimeError(f'{path} must be a '
                            f'{"or ".join(suffixes)} file or a directory.')


    path = args.output_path
    is_dir = path.is_dir() or path.suffix == ''
    if not (
        path.suffix == suffixes[0]
        or path.suffix == suffixes[2]
        or path.suffix == suffixes[3]
        or is_dir
    ):
        raise ValueError(
            f'{path} must be a {suffixes[0]}, {suffixes[2]} or {suffixes[3]} '
            'file or a directory.'
        )

    if is_dir:
        path = save_dir(path, data)
    elif path.suffix == suffixes[0]:
        path = save_npz(path, data)
    elif path.suffix == suffixes[2]:
        path = Dataset(**data).save_h5ad(path)
    else:
        net = NetworkParameter(
            data['basal'].size - 1,
            data.pop('gene_names'),
            data.pop('layout')
        )
        for name, value in data.items():
            getattr(net, name)[:] = value[:]
        path = net.save_json(path)

    print(path)

def add_subcommand(main_subparsers):
    parser = main_subparsers.add_parser(
        'convert',
        help='convert help'
    )

    parser.add_argument(
        'path', 
        type=Path,
        help='path to convert. ' 
             'It is a .npz file or a directory or a .txt (dataset).'
    )
    parser.add_argument(
        'output_path',
        type=Path,
        help='destination path. It is a .npz file or a directory.'
    )

    parser.set_defaults(run=convert)
