# from __future__ import annotations
from typing import Iterable, Dict, Any, Union

import numpy as np
import numpy.typing as npt
from pathlib import Path
from dataclasses import dataclass, astuple


@dataclass
class ParamInfos:
    required: bool
    dtype: npt.DTypeLike
    ndim: int

    def __iter__(self):
        return iter(astuple(self))


def _check_names(
    names: Iterable[str], 
    param_names: Dict[str, ParamInfos]
) -> None:
    cur_required_names: list[str] = []
    for name in names:
        if name not in param_names:
            raise RuntimeError(
                "Unexpected array name, "
                f"{name} is not in {list(param_names.keys())}."
            )
        elif param_names[name].required:
            cur_required_names.append(name)

    for name, infos in param_names.items():
        if infos.required and name not in cur_required_names:
            raise RuntimeError(f"{name} array is missing.")


def load_dir(
    path: Union[str, Path], 
    param_names: Dict[str, ParamInfos]
) -> Dict[str, npt.NDArray[Any]]:
    path = Path(path)  # convert it to Path (needed for str)
    if not path.exists():
        raise RuntimeError(f"{path} doesn't exist.")
    suffix = ".txt"
    _check_names(map(lambda p: p.stem, path.glob(f"*{suffix}")), param_names)

    data: dict[str, npt.NDArray[Any]] = {}
    for name, (required, dtype, ndim) in param_names.items():
        file_name = (path / name).with_suffix(suffix)
        if required or file_name.exists():
            data[name] = np.loadtxt(file_name, dtype=dtype, ndmin=ndim)

    return data


def load_npz(
    path: Union[str, Path], 
    param_names: Dict[str, ParamInfos]
) -> Dict[str, npt.NDArray[Any]]:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"{path} doesn't exist.")

    data: dict[str, npt.NDArray[Any]] = {}
    with np.load(path) as npz_file:
        data = dict(npz_file)

    _check_names(data.keys(), param_names)

    return data


def save_dir(
    path: Union[str, Path], 
    output_dict: Dict[str, npt.NDArray[Any]]
) -> Path:
    path = Path(path).with_suffix("")
    path.mkdir(parents=True, exist_ok=True)

    for key, value in output_dict.items():
        file_name = (path / key).with_suffix(".txt")
        if value.dtype == np.uint:
            max_val = np.max(value)
            width = 1 if max_val == 0 else int(np.log10(max_val) + 1.0)
            np.savetxt(file_name, value, fmt=f"%{width}d")
        elif value.dtype.type is np.str_:
            np.savetxt(file_name, value, fmt="%s")
        else:
            np.savetxt(file_name, value)

    return path


def save_npz(
    path: Union[str, Path], 
    output_dict: Dict[str, npt.NDArray[Any]]
) -> Path:
    path = Path(path).with_suffix(".npz")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output_dict)

    return path
