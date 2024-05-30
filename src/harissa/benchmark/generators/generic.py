from typing import (
    Dict, 
    List, 
    Generic, 
    TypeVar, 
    Union, 
    Optional
)

from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import make_archive, unpack_archive
from alive_progress import alive_bar

T = TypeVar('T')

class GenericGenerator(Generic[T]):
    def __init__(self,
        sub_directory_name: str, 
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        self._items : Optional[Dict[str, T]] = None
        self.sub_directory_name = sub_directory_name

        self.include = include or ['**']
        self.exclude = exclude or []
        
        if path is not None:
            self.load(path)


    def match(self, path: Optional[Union[str, Path]], suffix: str = ''):
        path = Path(path)

        return (
            any([path.match(f'{p}{suffix}') for p in self.include]) 
            and all([not path.match(f'{p}{suffix}') for p in self.exclude])
        )
    
    def match_rec(self, path: Optional[Union[str, Path]]):
        def add_paths(p:Optional[Path], acc: List[Path]):
            if p is not None:
                if p.is_dir():
                    for sub_p in p.iterdir():
                        acc = add_paths(sub_p, acc)
                elif self.match(p.relative_to(path), '.npz'):
                    acc = add_paths(None, acc + [p])
            
            return acc
            
        return add_paths(Path(path), [])
    
    @property
    def items(self) -> Dict[str, T]:
        self.generate()
        return self._items

    def load(self, path: Union[str, Path]) -> None:        
        def __load(p : Path):
            p = p / self.sub_directory_name
            if not p.is_dir():
                raise ValueError(f'{p} must be an existing directory.')
        
            try:
                self._load(p)
            except BaseException as e:
                self._items = None
                raise e
            
        path = Path(path)
        if path.suffix != '':
            with TemporaryDirectory() as tmp_dir:
                unpack_archive(path, tmp_dir)
                __load(Path(tmp_dir))
        else:
            __load(path)
    
    def generate(self, force_generation: bool = False) -> None:
        if self._items is None or force_generation:
            try:
                self._generate()
            except BaseException as e:
                self._items = None
                raise e


    def save(self, 
        path: Union[str, Path], 
        archive_format: Optional[str] = None
    ) -> Path:
        self.generate()

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
    
    def _load(self, path: Path) -> None:
        raise NotImplementedError
    
    def _generate(self) -> None:
        raise NotImplementedError

    def _save(self, path: Path) -> None:
        raise NotImplementedError