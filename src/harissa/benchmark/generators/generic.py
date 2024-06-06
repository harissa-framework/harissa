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
from tempfile import TemporaryDirectory, mkdtemp
from shutil import make_archive, unpack_archive, rmtree
from alive_progress import alive_bar

K = TypeVar('K', str, Tuple[Union[str, int],...])
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

        self.include = include or ['**']
        self.exclude = exclude or []
        
        self.path = path
        self.verbose = verbose


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
    
    def _sub_dir(self, path: Path) -> Path:
        p = path / self.sub_directory_name
        if not p.is_dir():
            raise ValueError(f'{p} must be an existing directory.')
        
        return p
    
    def _check_path(self) -> Path:
        self.path = Path(self.path)
        if self.path.suffix != '':
            path = mkdtemp()
            unpack_archive(self.path, path)
        else:    
            path = self.path

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
            return self._load(self._check_path())
        else:
            return self._generate()


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
    
    def keys(self) -> Iterator[K]:
        if self.path is not None:
            self.path = Path(self.path)
            if self.path.suffix != '':
                path = mkdtemp()
                unpack_archive(self.path, path)
            else:    
                path = self.path
            
            sub_dir = path / self.sub_directory_name
            if not sub_dir.is_dir():
                raise ValueError(f'{sub_dir.name} is missing from {path}.')
        else:
            path = None

        return self._keys(path)
    
    def _load(self, path: Path) -> Iterator[Tuple[K, V]]:
        raise NotImplementedError
    
    def _generate(self) -> Iterator[Tuple[K, V]]:
        raise NotImplementedError

    def _save(self, path: Path) -> None:
        raise NotImplementedError
    
    def _keys(self, path: Optional[Path]) -> Iterator[K]:
        raise NotImplementedError