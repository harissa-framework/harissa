from typing import (
    Dict, 
    List,
    Tuple, 
    TypeVar, 
    Union, 
    Optional,
    Callable,
    Generic
)

from collections.abc import Iterator

from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import make_archive, unpack_archive
from harissa.utils.progress_bar import alive_bar

K = TypeVar('K', str, Tuple[str,...])
V = TypeVar('V')

def _to_str(key: K) -> str:
    """
    Transform a key to a path-like string.

    Parameters
    ----------
    key
        key to be transformed
   
    """
    if isinstance(key, str):
        return key
    else:
        return str(Path().joinpath(*key))

class GenericGenerator(Generic[K, V]):
    """
    Generic abstract class for benchmark generators

    Attributes
    ----------
    verbose: bool
        If True display a progress bar during the generation.

    """
    def __init__(self,
        sub_directory_name: str, 
        include: List[K] = ['*'],
        exclude: List[K] = [],
        path: Optional[Union[str, Path]] = None,
        verbose: bool = None
    ) -> None:
        self._sub_directory_name = sub_directory_name

        self.include = include
        self.exclude = exclude
        self.path = path
        self.verbose = verbose

    @property
    def path(self) -> Optional[Path]:
        """
        Path where to load values. 
        If path is None, values are generated on the fly.
        """
        return self._path
    
    @path.setter
    def path(self, path: Optional[Union[str, Path]]):
        if path is not None:
            if not isinstance(path, (str, Path)):
                raise TypeError('path must be an str or a Path.')
            self._set_path(Path(path))
        else:
            self._set_path(None)

    def _set_path(self, path: Optional[Path]):
        if path is not None and path.is_dir():
            self._path = self._check_path(path)
        else:
            self._path = path

    @property
    def include(self) -> List[K]:
        """
        List of keys to be included in the generation. 
        The wildcard ``*`` can be used. 
        See https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.match
        """
        return self._include
    
    @include.setter
    def include(self, include) -> None:
        if not isinstance(include, list):
            raise TypeError('include must be a list of keys.')
        self._include = include

    @property
    def exclude(self) -> List[K]:
        """
        List of keys to be excluded in the generation. 
        The wildcard ``*`` can be used. 
        See https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.match

        """
        return self._exclude
    
    @exclude.setter
    def exclude(self, exclude):
        if not isinstance(exclude, list):
            raise TypeError('exclude must be a list of keys.')
        self._exclude = exclude

    def _to_path(self, key: K, path: Optional[Path] = None) -> Path:
        """
        Transform a key to a path.

        Parameters
        ----------
        key
            key to be transformed
        path
            root of the returned path
        """
        if path is None:
            path = self.path or Path()

        return path.joinpath(self._sub_directory_name, _to_str(key))

    def match(self, key: K) -> bool:
        """
        Test if a key is inside `self.include` and outside of `self.exclude`.

        Parameters
        ----------
        key
            key to be tested

        """
        path = Path(_to_str(key))
        include = map(_to_str, self.include)
        exclude = map(_to_str, self.exclude)

        return (
            any([path.match(pattern) for pattern in include]) 
            and all([not path.match(pattern) for pattern in exclude])
        )
    
    def as_dict(self) -> Dict[K, V]:
        """
        Return self as a dictionary (values stored in memory).
        """
        return dict(iter(self))
    
    def _check_path(self, path) -> Path:
        sub_dir = path / self._sub_directory_name
        if not sub_dir.is_dir():
            raise ValueError(f'{sub_dir.name} is missing from {path}.')
        
        return path
    
    def __getitem__(self, key: K) -> V:
        """
        return the value mapped to the key

        Parameters
        ----------
        key
            input key

        Raises
        ------
        KeyError
        
        """
        if self.match(key):
            if self.path is not None:
                if self.path.suffix != '':
                    old_path = self.path
                    with TemporaryDirectory() as tmp_dir:
                        unpack_archive(self.path, tmp_dir)
                        self.path = tmp_dir
                        value = self._load_value(key)
                    self.path = old_path
                else:
                    value = self._load_value(key) 
            else:
                value = self._generate_value(key)

            return value
        else:
            raise KeyError
        
    def save_item(self, 
        path: Union[str, Path], 
        key: K, 
        value: Optional[V] = None
    ) -> None:
        """
        save value in the corresponding sub directory.

        Parameters
        ----------
        path
            path where the value is saved.
        key
            input key
        value
            input value. 
            If value is None then value is generated from the key.
        """
        if value is None:
            value = self[key]

        self._save_item(Path(path), (key, value))
    
    def __iter__(self) -> Iterator[K]:
        """
        return an iterator on keys
        """
        yield from self.keys()
    
    def __len__(self) -> int:
        count = 0
        for _ in self.keys():
            count += 1

        return count

    def keys(self) -> Iterator[K]:
        """
        return an iterator on keys
        """
        yield from self._generate(None)

    def items(self) -> Iterator[Tuple[K, V]]:
        """
        return an iterator on items

        Yields
        ------
        Tuple[K, V]
            _description_
        """
        yield from self._generate(lambda key, value: (key, value))

    def values(self) -> Iterator[V]:
        yield from self._generate(lambda key, value: value)

    def _generate(self, 
        projection_fn: Optional[Callable[[K, V], Union[K, Tuple[K, V]]]]
    ) -> Union[Iterator[K], Iterator[K, V], Iterator[V]]:
        generate_keys_only = projection_fn is None
        if not generate_keys_only:
            def yield_projected_items(value_fn, title, keys):
                with alive_bar(
                    len(keys), 
                    title=title, 
                    disable=not self.verbose
                ) as bar:
                    for key in keys:
                        bar.text(' - '.join(key))
                        value = value_fn(key)
                        bar()
                        yield projection_fn(key, value)
                
        if self.path is not None:
            title = f'Loading {self._sub_directory_name}'
            if self.path.suffix != '':
                old_path = self.path
                with TemporaryDirectory() as tmp_dir:
                    unpack_archive(self.path, tmp_dir)
                    self.path = tmp_dir
                    if generate_keys_only:
                        yield from self._load_keys()
                    else:
                        yield from yield_projected_items(
                            self._load_value,
                            title,
                            list(self._load_keys())
                        )
                self.path = old_path
            else:
                if generate_keys_only:
                    yield from self._load_keys()
                else:
                    yield from yield_projected_items(
                        self._load_value,
                        title,
                        list(self._load_keys())
                    )
        else:
            if generate_keys_only:
                yield from self._generate_keys()
            else:
                yield from yield_projected_items(
                    self._generate_value,
                    f'Generating {self._sub_directory_name}',
                    list(self._generate_keys())
                )

    def save(self, 
        path: Union[str, Path], 
        archive_format: Optional[str] = None
    ) -> Path:
        path = Path(path).with_suffix('')
        if archive_format is not None:
            with TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                for item in self.items():
                    self._save_item(tmp_path, item)
                with alive_bar(
                    title='Archiving', 
                    monitor=False,
                    stats= False
                ) as bar:
                    path=Path(make_archive(str(path), archive_format, tmp_dir))
                    bar()
        else:
            path.mkdir(parents=True, exist_ok=True)
            for item in self.items():
                self._save_item(path, item)
            
        return path.absolute()

    def _load_value(self, key: K) -> V:
        raise NotImplementedError
    
    def _generate_value(self, key: K) -> V:
        raise NotImplementedError
    
    def _load_keys(self) -> Iterator[K]:
        root = self.path / self._sub_directory_name
        def yield_rec(p: Path):
            if p.is_dir():
                for sub_p in p.iterdir():
                    yield from yield_rec(sub_p)
            else:
                key = str(p.relative_to(root).with_suffix(''))
                if self.match(key):
                    yield key

        yield from yield_rec(root)
    
    def _generate_keys(self) -> Iterator[K]:
        raise NotImplementedError
    
    def _save_item(self, path: Path, item: Tuple[K, V]):
        raise NotImplementedError