import numpy as np
from pathlib import Path

from harissa import NetworkParameter
from harissa.simulation import Simulation

export_format = ('npz', 'txt')

def load(path: Path, param_names: dict | None = None) -> dict:
    data = None
    suffixes = tuple(map(lambda f: f'.{f}', export_format))
    if path.suffix == suffixes[0]:
        with np.load(path) as npz_file:
            data = dict(npz_file)
    elif path.is_dir():
        data = {}
        if param_names is None:
            file_list = path.glob(f'*{suffixes[1]}')
            for file_name in file_list:
                data[str(file_name.with_suffix(''))] = np.loadtxt(
                    path / file_name
                )
        else:
            for name, required in param_names.items():
                file_name = (path / name).with_suffix(suffixes[1])
                if required or file_name.exists():
                    data[name] = np.loadtxt(file_name)
    else:
        raise RuntimeError(f'{path} must be a .npz file or a directory.')

    return data

def load_network_parameter(path: Path) -> NetworkParameter:
    network_param_names = {
        'burst_frequency_min': True,
        'burst_frequency_max': True,
        'burst_size_inv': True,
        'creation_rna': True,
        'creation_protein': True,
        'degradation_rna': True,
        'degradation_protein': True,
        'basal': True,
        'interaction': True
    }
    data = load(path, network_param_names)
    network_param = NetworkParameter(data['basal'].size - 1)

    for key, value in data.items():
        getattr(network_param, key)[:] = value[:]

    return network_param

def load_simulation_parameter(path: Path, burn_in: float| None) -> dict:
    sim_param_names = {
        'time_points': True,
        'M0': False,
        'P0': False
    }
    sim_param = load(path, sim_param_names)
    sim_param['burn_in'] = burn_in

    return sim_param


def save(output_path: Path, output_dict: dict, output_format: str) -> None:
    if output_format not in export_format:
        raise RuntimeError(f'{output_format} must be npz '
                           f'{"or ".join(export_format)}')

    if output_format == export_format[0]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **output_dict)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        for key, value in output_dict.items():
            file_name = (output_path / key).with_suffix(f'.{export_format[1]}')
            if value.dtype == np.uint:
                max_val = np.max(value)
                width = 1 if max_val == 0 else int(np.log10(max_val) + 1.0) 
                np.savetxt(file_name, value, fmt=f'%{width}d')
            else:
                np.savetxt(file_name, value)


def save_network_parameter(output_path: Path, 
                           network_parameter: NetworkParameter, 
                           output_format: str) -> None:
    save(
        output_path, 
        {
            'burst_frequency_min': network_parameter.burst_frequency_min,
            'burst_frequency_max': network_parameter.burst_frequency_max,
            'burst_size_inv': network_parameter.burst_size_inv,
            'creation_rna': network_parameter.creation_rna,
            'creation_protein': network_parameter.creation_protein,
            'degradation_rna': network_parameter.degradation_rna,
            'degradation_protein': network_parameter.degradation_protein,
            'basal': network_parameter.basal,
            'interaction': network_parameter.interaction 
        }, 
        output_format
    )

def save_simulation_result(output_path: Path, 
                           result: Simulation.Result, 
                           output_format:str) -> None:
    save(
        output_path, 
        {
            'time_points': result.time_points,
            'rna_levels': result.rna_levels,
            'protein_levels': result.protein_levels
        },
        output_format
    )
    

def convert(path: Path) -> None: 
    save(path, load(path), export_format[int(path.is_dir())])