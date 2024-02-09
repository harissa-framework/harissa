import numpy as np
from pathlib import Path
from dataclasses import asdict

from harissa import NetworkParameter
from harissa.simulation import Simulation
from harissa.dataset import Dataset

suffixes = ('.npz', '.txt')

names_allowed = {
    'dataset': {
        'time_points': (True, np.float_), 
        'count_matrix': (True, np.uint),
        # 'gene_names': (False, np.str_)
    },
    'network parameter': {
        'burst_frequency_min': (True, np.float_),
        'burst_frequency_max': (True, np.float_),
        'burst_size_inv': (True, np.float_),
        'creation_rna': (True, np.float_),
        'creation_protein': (True, np.float_),
        'degradation_rna': (True, np.float_),
        'degradation_protein': (True, np.float_),
        'basal': (True, np.float_),
        'interaction': (True, np.float_)
    },
    'simulation parameter': {
        'time_points': (True, np.float_),
        'M0': (False, np.float_),
        'P0': (False, np.float_)
    },
    'simulation result': {
        'time_points': (True, np.float_),
        'rna_levels': (True, np.float_),
        'protein_levels': (True, np.float_)
    }  
}

def load_txt(path: str | Path) -> dict:
    # Backward compatibility, dataset inside a txt file.
    # It assumes that the 1rst column is the time points (arr_list[0]) 
    # and the rest is the count matrix (arr_list[1])
    data = np.loadtxt(path)
    data_list = [data[:, 0].copy(), data.astype(np.uint)]
    # data_list = [data[:, 0].copy(), data.astype(np.uint), None]

    # Set stimuli instead of time_points
    data_list[1][:, 0] = data_list[0] != 0.0
    data_dict = {}
    for i, name in enumerate(names_allowed['dataset']):
        data_dict[name] = data_list[i]

    return data_dict

def _check_names(names, param_names) -> None:
    cur_required_names = []
    for name in names:
        if name not in param_names:
            raise RuntimeError('Unexpected array name, '
                                f'{name} is not in {param_names}.')
        elif param_names[name][0]:
            cur_required_names.append(name)

    for name, (required, _) in param_names.items():
        if required and name not in cur_required_names:
            raise RuntimeError(f'{name} array is missing')

def load_dir(path : str | Path, param_names: dict) -> dict:
    path = Path(path) # convert it to Path (needed for str)
    suffix = suffixes[1]
    _check_names(map(lambda p : p.stem, path.glob(f'*{suffix}')), param_names)

    data = {}
    for name, (required, dtype) in param_names.items():
        file_name = (path / name).with_suffix(suffix)
        if required or file_name.exists():
            data[name] = np.loadtxt(file_name, dtype=dtype)
    
    return data


def load_npz(path: str | Path, param_names: dict) -> dict:
    data = {}
    with np.load(path) as npz_file:
        data = dict(npz_file)

    _check_names(data.keys(), param_names)

    return data

def load_dataset_txt(path: str | Path) -> Dataset:
    path = Path(path) # convert it to Path (needed for str)
    
    if path.suffix == suffixes[1]:
        data = load_txt(path)
    else:
        data = load_dir(path, names_allowed['dataset'])

    return Dataset(**data)

def load_dataset(path: str | Path) -> Dataset:
    return Dataset(**load_npz(path, names_allowed['dataset']))


def _create_load_parameter(load_fn):
    def load_network_parameter(path: str | Path) -> NetworkParameter:
        data = load_fn(path, names_allowed['network parameter'])
        network_param = NetworkParameter(data['basal'].size - 1)

        for key, value in data.items():
            getattr(network_param, key)[:] = value[:]

        return network_param
    
    return load_network_parameter

load_network_parameter_txt = _create_load_parameter(load_dir)
load_network_parameter = _create_load_parameter(load_npz)

def _create_load_simulation_parameter(load_fn):
    def load_simulation_parameter(path: str | Path, 
                                  burn_in: float| None) -> dict:
        sim_param = load_fn(path, names_allowed['simulation parameter'])
        sim_param['burn_in'] = burn_in
        sim_param['time_points'] = np.unique(sim_param['time_points'])

        return sim_param
    
    return load_simulation_parameter

load_simulation_parameter_txt = _create_load_simulation_parameter(load_dir)
load_simulation_parameter = _create_load_simulation_parameter(load_npz)

def save_dir(path: str | Path, output_dict: dict) -> Path:
    path = Path(path).with_suffix('')
    path.mkdir(parents=True, exist_ok=True)

    for key, value in output_dict.items():
        file_name = (path / key).with_suffix(suffixes[1])
        if value.dtype == np.uint:
            max_val = np.max(value)
            width = 1 if max_val == 0 else int(np.log10(max_val) + 1.0) 
            np.savetxt(file_name, value, fmt=f'%{width}d')
        else:
            np.savetxt(file_name, value)

    return path
def save_npz(path: str | Path, output_dict: dict) -> Path:
    path = Path(path).with_suffix(suffixes[0])

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output_dict)
    
    return path

def _create_save_network_parameter(save_fn):
    def save_network_parameter(path: str | Path, 
                               network_parameter: NetworkParameter) -> Path:
        return save_fn(
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
            }
        )
    
    return save_network_parameter

save_network_parameter_txt = _create_save_network_parameter(save_dir)
save_network_parameter = _create_save_network_parameter(save_npz)

def _create_save_simulation_result(save_fn):
    def save_simulation_result(path: str | Path, 
                               result: Simulation.Result) -> Path:
        return save_fn(path, asdict(result))
    
    return save_simulation_result

save_simulation_result_txt = _create_save_simulation_result(save_dir)
save_simulation_result = _create_save_simulation_result(save_npz)

def _create_save_dataset(save_fn):
    def save_dataset(path: str | Path, dataset: Dataset) -> Path:
        return save_fn(path, asdict(dataset))
    
    return save_dataset

save_dataset_txt = _create_save_dataset(save_dir)
save_dataset = _create_save_dataset(save_npz)
    

def convert(path: str | Path, output_path: str | Path | None = None) -> Path: 
    path = Path(path)
    is_dir = path.is_dir()
    if path.suffix == suffixes[1]:
        data = load_txt(path)
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

    if output_path is not None:
        path = Path(output_path)
        is_dir = path.is_dir()
        if not (path.suffix == suffixes[0] or is_dir):
            raise ValueError(f'{path} must be a '
                             f'{suffixes[0]} file or a directory.')
        
    return save_npz(path, data) if is_dir else save_dir(path, data)
    