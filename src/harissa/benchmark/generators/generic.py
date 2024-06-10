from typing import (
    Dict, 
    List,
    Tuple, 
    TypeVar, 
    Union, 
    Optional
)

from collections.abc import Iterator, Iterable

from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import make_archive, unpack_archive, rmtree
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

    def match(self, path: Union[str, Path], suffix: str = ''):
        path = Path(path)

        return (
            any([path.match(f'{p}{suffix}') for p in self.include]) 
            and all([not path.match(f'{p}{suffix}') for p in self.exclude])
        )
    
    def match_rec(self, path: Union[str, Path]):
        def add_paths(p:Optional[Path], acc: List[Path]):
            if p is not None:
                if p.is_dir():
                    for sub_p in p.iterdir():
                        acc = add_paths(sub_p, acc)
                elif self.match(p.relative_to(path), '.npz'):
                    acc = add_paths(None, acc + [p])
            
            return acc
            
        return add_paths(Path(path) / self.sub_directory_name, [])
    
    def as_dict(self) -> Dict[K, V]:
        return dict(iter(self))
    
    def _check_path(self, path) -> Path:
        sub_dir = path / self.sub_directory_name
        if not sub_dir.is_dir():
            raise ValueError(f'{sub_dir.name} is missing from {path}.')
        
        return path
    
    def remove_tmp_dir(self, tmp_dir):
        if self.path != tmp_dir:
            print(f'deleting tmp_dir {tmp_dir}')
            rmtree(tmp_dir)
    
    def __iter__(self) -> Iterator[Tuple[K, V]]:
        if self.path is not None: 
            if self.path.suffix != '':
                with TemporaryDirectory() as tmp_dir:
                    unpack_archive(self.path, tmp_dir)
                    yield from self._load(self._check_path(tmp_dir))
            else:
                yield from self._load(self._check_path(self.path))
        else:
            yield from self._generate()
        
    def keys(self) -> Iterator[K]:
        if self.path is not None:
            if self.path.suffix != '':
                with TemporaryDirectory() as tmp_dir:
                    unpack_archive(self.path, tmp_dir)
                    yield from self._load_keys(self._check_path(tmp_dir))
            else:
                yield from self._load_keys(self._check_path(self.path)) 
        else:
            yield from self._generate_keys()

    def save(self, 
        path: Union[str, Path], 
        archive_format: Optional[str] = None
    ) -> Path:
        path = Path(path).with_suffix('')
        if archive_format is not None:
            with TemporaryDirectory() as tmp_dir:
                self._save(Path(tmp_dir) / self.sub_directory_name)
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
            self._save(output)

        return path.absolute()
    
    def __len__(self) -> int:
        count = 0
        for _ in self.keys():
            count += 1

        return count
    
    def _load_keys(self, path: Path) -> Iterator[K]:
        raise NotImplementedError
    
    def _load(self, path: Path) -> Iterator[Tuple[K, V]]:
        raise NotImplementedError
    
    def _generate_keys(self) -> Iterator[K]:
        raise NotImplementedError

    def _generate(self) -> Iterator[Tuple[K, V]]:
        raise NotImplementedError

    def _save(self, path: Path) -> None:
        raise NotImplementedError