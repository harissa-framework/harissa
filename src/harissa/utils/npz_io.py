import numpy as np
from pathlib import Path

def _check_names(names, param_names) -> None:
    cur_required_names = []
    for name in names:
        if name not in param_names:
            raise RuntimeError('Unexpected array name, '
                              f'{name} is not in {list(param_names.keys())}.')
        elif param_names[name][0]:
            cur_required_names.append(name)

    for name, (required, _) in param_names.items():
        if required and name not in cur_required_names:
            raise RuntimeError(f'{name} array is missing.')

def load_dir(path : str | Path, param_names: dict) -> dict:
    path = Path(path) # convert it to Path (needed for str)
    if not path.exists():
        raise RuntimeError(f"{path} doesn't exist.")
    suffix = '.txt'
    _check_names(map(lambda p : p.stem, path.glob(f'*{suffix}')), param_names)

    data = {}
    for name, (required, dtype) in param_names.items():
        file_name = (path / name).with_suffix(suffix)
        if required or file_name.exists():
            data[name] = np.loadtxt(file_name, dtype=dtype)
    
    return data


def load_npz(path: str | Path, param_names: dict) -> dict:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"{path} doesn't exist.")
    
    data = {}
    with np.load(path) as npz_file:
        data = dict(npz_file)

    _check_names(data.keys(), param_names)

    return data

def save_dir(path: str | Path, output_dict: dict) -> Path:
    path = Path(path).with_suffix('')
    path.mkdir(parents=True, exist_ok=True)

    for key, value in output_dict.items():
        file_name = (path / key).with_suffix('.txt')
        if value.dtype == np.uint:
            max_val = np.max(value)
            width = 1 if max_val == 0 else int(np.log10(max_val) + 1.0) 
            np.savetxt(file_name, value, fmt=f'%{width}d')
        elif value.dtype.type is np.str_:
            np.savetxt(file_name, value, fmt='%s')
        else:
            np.savetxt(file_name, value)

    return path
def save_npz(path: str | Path, output_dict: dict) -> Path:
    path = Path(path).with_suffix('.npz')

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output_dict)
    
    return path