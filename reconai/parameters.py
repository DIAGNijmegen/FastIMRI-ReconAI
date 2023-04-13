from dataclasses import dataclass, InitVar
from pathlib import Path
from datetime import datetime
from typing import List

from .config import load, load_str, Config


model_package = 'reconai.models.'

@dataclass
class Parameters:
    in_dir: Path
    out_dir: Path
    model: str
    _config: InitVar[Path]
    debug: bool = False

    def __post_init__(self, _config: Path):
        self.date = datetime.now().strftime("%Y%m%d_%H%M")
        self.name = '_'.join([_config.name if _config else 'default', self.model] + (['debug'] if self.debug else []))
        self.model = model_package + self.model

        self._yaml = load(_config) if _config else load_str()
        self.config = Config(self._yaml.data)

    def as_yaml(self):
        return self._yaml.as_yaml()

    @property
    def name_date(self) -> str:
        return f'{self.name}_{self.date}'

    @staticmethod
    def model_names(*args: str) -> List[str]:
        names = []
        for model in args:
            if not model.startswith(model_package):
                raise ValueError(f'Model must be contained in the {model_package} package')
            names.append(model.split('.')[-1])
        return names
