import numpy as np
from pathlib import Path

from harissa import NetworkParameter
from harissa.simulation import Simulation
from harissa.dataset import Dataset

export_format = ('npz', 'txt')

def load(path: str | Path, param_names: dict | None = None) -> dict:
    path = Path(path) # convert it to Path (needed for str)
    if not path.exists():
        raise RuntimeError(f"{path} doesn't exist.")
    
    data = {}
    suffixes = tuple(map(lambda f: f'.{f}', export_format))
    if path.suffix == suffixes[0]:
        with np.load(path) as npz_file:
            data = dict(npz_file)
    elif path.suffix == suffixes[1]:
        # Backward compatibility, dataset inside a txt file.
        # It assumes that the 1rst column is the time points (arr_list[0]) 
        # and the rest is the count matrix (arr_list[1])
        data_real = np.loadtxt(path)
        arr_list = [data_real[:, 0].copy(), data_real.astype(np.uint)]
        # Set stimuli instead of time_points
        arr_list[1][:, 0] = arr_list[0] != 0.0
        param_names = param_names or ('time_points', 'count_matrix')
        for i, name in enumerate(param_names):
            data[name] = arr_list[i]
    elif path.is_dir():
        if param_names is None:
            file_list = path.glob(f'*{suffixes[1]}')
            for file_name in file_list:
                data[file_name.stem] = np.loadtxt(file_name)
        else:
            for name, required in param_names.items():
                file_name = (path / name).with_suffix(suffixes[1])
                if required or file_name.exists():
                    data[name] = np.loadtxt(file_name)
    else:
        raise RuntimeError(f'{path} must be a .npz file or a directory.')

    return data

def load_dataset(path: str | Path) -> Dataset:
    return Dataset(**load(path, {'time_points': True, 'count_matrix': True}))

def load_network_parameter(path: str | Path) -> NetworkParameter:
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

def load_simulation_parameter(path: str | Path, burn_in: float| None) -> dict:
    sim_param_names = {
        'time_points': True,
        'M0': False,
        'P0': False
    }
    sim_param = load(path, sim_param_names)
    sim_param['burn_in'] = burn_in

    return sim_param


def save(path: str | Path, output_dict: dict, output_format: str) -> Path:
    path = Path(path) # convert it to Path (needed for str)
    if output_format not in export_format:
        raise ValueError(f'{output_format} must be '
                         f'{"or ".join(export_format)}')

    if output_format == export_format[0]:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **output_dict)
        path = path.with_suffix(f'.{export_format[0]}')
    else:
        path.mkdir(parents=True, exist_ok=True)
        for key, value in output_dict.items():
            file_name = (path / key).with_suffix(f'.{export_format[1]}')
            if value.dtype == np.uint:
                max_val = np.max(value)
                width = 1 if max_val == 0 else int(np.log10(max_val) + 1.0) 
                np.savetxt(file_name, value, fmt=f'%{width}d')
            else:
                np.savetxt(file_name, value)

    return path

def save_network_parameter(path: str | Path, 
                           network_parameter: NetworkParameter, 
                           output_format: str) -> Path:
    return save(
        path, 
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

def save_simulation_result(path: str | Path, 
                           result: Simulation.Result, 
                           output_format:str) -> Path:
    return save(
        path, 
        {
            'time_points': result.time_points,
            'rna_levels': result.rna_levels,
            'protein_levels': result.protein_levels
        },
        output_format
    )
    

def convert(path: str | Path, output_path: str | Path | None = None) -> Path: 
    path = Path(path)
    data = load(path)
    if output_path is None:
        return save(
            path.with_suffix(''), 
            data, 
            export_format[1 - path.is_dir()]
        )
    else:
        output_path = Path(output_path)
        txt_suffix = f'.{export_format[1]}'
        if output_path.suffix == txt_suffix:
            raise ValueError(f'{output_path} must be npz or a directory.')
        
        suffix = output_path.suffix or txt_suffix
        return save(
            output_path.with_suffix(''),
            data,
            suffix[1:]
        )