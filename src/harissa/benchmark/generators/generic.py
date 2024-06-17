from typing import (
    Dict, 
    List,
    Tuple, 
    TypeVar, 
    Union, 
    Optional,
    Callable
)

from collections.abc import Iterator, Iterable

from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import make_archive, unpack_archive
from alive_progress import alive_bar

K = TypeVar('K', str, Tuple[str,...])
V = TypeVar('V')
class GenericGenerator(Iterable[Tuple[K, V]]):
    def __init__(self,
        sub_directory_name: str, 
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        path: Optional[Union[str, Path]] = None,
        verbose: bool = False
    ) -> None:
        self.sub_directory_name = sub_directory_name
        self.verbose = verbose

        self._include = ['**']
        self._exclude = []
        
        self.path = path

        if include is not None:
            self.include = include
        if exclude is not None:
            self.exclude = exclude

    @property
    def path(self) -> Optional[Path]:
        return self._path
    
    @path.setter
    def path(self, path: Optional[Union[str, Path]]):
        if path is not None:
            if not isinstance(path, (str, Path)):
                raise TypeError('path must be an str or a Path.')
            self.set_path(Path(path))
        else:
            self._path = None


    def set_path(self, path: Path):
        self._path = path

    @property
    def verbose(self) -> bool:
        return self._verbose
    
    @verbose.setter
    def verbose(self, verbose: bool):
        if not isinstance(verbose, bool):
            raise TypeError('verbose must be a boolean.')
        self.set_verbose(verbose)

    def set_verbose(self, verbose:bool):
        self._verbose = verbose

    @property
    def include(self):
        return self._include    
    @include.setter
    def include(self, include):
        if not isinstance(include, list):
            raise TypeError('include must be a list of str.')
        self.set_include(include)

    def set_include(self, include):
        self._include = include

    @property
    def exclude(self):
        return self._exclude
    
    @exclude.setter
    def exclude(self, exclude):
        if not isinstance(exclude, list):
            raise TypeError('exclude must be a list of str.')
        self.set_exclude(exclude)

    def set_exclude(self, exclude):
        self._exclude = exclude

    def match(self, key: K):
        def to_str(k: K) -> str:
            return k if isinstance(k, str) else str(Path().joinpath(*k))

        path = Path(to_str(key))
        include = map(to_str, self.include)
        exclude = map(to_str, self.exclude)

        return (
            any([path.match(pattern) for pattern in include]) 
            and all([not path.match(pattern) for pattern in exclude])
        )
    
    def as_dict(self) -> Dict[K, V]:
        return dict(iter(self))
    
    def _check_path(self, path) -> Path:
        sub_dir = path / self.sub_directory_name
        if not sub_dir.is_dir():
            raise ValueError(f'{sub_dir.name} is missing from {path}.')
        
        return path
    
    def __getitem__(self, key: K) -> V:
        if self.match(key):
            if self.path is not None:
                if self.path.suffix != '':
                    with TemporaryDirectory() as tmp_dir:
                        unpack_archive(self.path, tmp_dir)
                        return self._load_value(
                            self._check_path(Path(tmp_dir)),
                            key
                        )
                else:
                    return self._load_value(self._check_path(self.path), key) 
            else:
                return self._generate_value(key)
        else:
            raise KeyError
    
    def __iter__(self) -> Iterator[K]:
        yield from self.keys()
    
    def __len__(self) -> int:
        count = 0
        for _ in self.keys():
            count += 1

        return count

    def keys(self) -> Iterator[K]:
        yield from self._generate(None)

    def items(self) -> Iterator[Tuple[K, V]]:
        yield from self._generate(lambda key, value: (key, value))

    def values(self) -> Iterator[V]:
        yield from self._generate(lambda key, value: value)

    def _generate(self, 
        projection_fn: Optional[Callable[[K, V], Union[K, Tuple[K, V]]]]
    ) -> Union[Iterator[K], Iterator[K, V], Iterator[V]]:
        generate_keys_only = projection_fn is None
        if not generate_keys_only:
            def yield_projected_items(value_fn, title, keys):
                self._pre_generate()
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
                self._post_generate()

        if self.path is not None:
            title = f'Loading {self.sub_directory_name}'
            if self.path.suffix != '':
                with TemporaryDirectory() as tmp_dir:
                    unpack_archive(self.path, tmp_dir)
                    tmp_path = self._check_path(Path(tmp_dir))
                    if generate_keys_only:
                        yield from self._load_keys(tmp_path)
                    else:
                        yield from yield_projected_items(
                            lambda key: self._load_value(tmp_path, key),
                            title,
                            list(self._load_keys(tmp_path))
                        )
            else:
                path = self._check_path(self.path)
                if generate_keys_only:
                    yield from self._load_keys(path)
                else:
                    yield from yield_projected_items(
                        lambda key: self._load_value(path, key),
                        title,
                        list(self._load_keys(path))
                    )
        else:
            if generate_keys_only:
                yield from self._generate_keys()
            else:
                yield from yield_projected_items(
                    self._generate_value,
                    f'Generating {self.sub_directory_name}',
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
                self._pre_save(tmp_path)
                self._save(tmp_path / self.sub_directory_name)
                self._post_save()
                with alive_bar(
                    title='Archiving', 
                    monitor=False, 
                    stats= False
                ) as bar:
                    path=Path(make_archive(str(path), archive_format, tmp_dir))
                    bar()
        else:
            output = path / self.sub_directory_name
            output.mkdir(parents=True, exist_ok=True)
            self._pre_save(path)
            self._save(output)
            self._post_save()

        return path.absolute()
    
    def _pre_save(self, path: Path):
        pass

    def _post_save(self):
        pass 
    
    def _pre_generate(self):
        pass

    def _post_generate(self):
        pass

    def _load_value(self, path: Path, key: K) -> V:
        raise NotImplementedError
    
    def _generate_value(self, key: K) -> V:
        raise NotImplementedError
    
    def _load_keys(self, path: Path) -> Iterator[K]:
        root = path / self.sub_directory_name
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

    def _save(self, path: Path) -> None:
        raise NotImplementedError