import importlib
from dataclasses import dataclass, field, InitVar, is_dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path

from strictyaml import load as strict_load, YAML


@dataclass
class Parameters:
    @dataclass
    class Data:
        """Parameters related to the dataset

        :param split_regex: first capture group is how to split mhas into cases (eg. '.*_(.*)_')
        :param filter_regex: result _is_ loaded
        :param shape_x: image columns to crop or zero-fill to
        :param shape_y: image rows to crop or zero-fill to
        :param sequence_length: length of T
        :param sequence_seed: seed for sequence generator
        :param normalize: image normalize divisor
        :param undersampling: how many k-lines to synthetically remove
        :param mask_seed: seed to Gaussian random k-space masks
        """
        split_regex: str = '.*_(.*)_'
        filter_regex: str = 'sag'
        shape_x: int = 256
        shape_y: int = 256
        sequence_length: int = 5
        sequence_seed: int = 11
        normalize: float = 1961.06
        undersampling: int = 8
        mask_seed: int = 11

    @dataclass
    class Model:
        """Parameters related to the model

        :param iterations: CRNN iterations
        :param filters: CRNN filter count
        :param kernelsize: CRNN kernel convolution size
        :param channels: CRNN channel width
        :param layers: CRNN total layers
        :param bcrnn: whether to include the BCRNN layer (False, to replace with regular CRNN layer)
        """
        iterations: int = 5
        filters: int = 64
        kernelsize: int = 3
        channels: int = 1
        layers: int = 5
        bcrnn: bool = True

    @dataclass
    class Train:
        """Parameters related to the model

        :param epochs: number of epochs
        :param folds: number of folds
        :param lr: learning rate
        :param lr_gamma: learning rate decay per epoch
        :param lr_decay_end: set lr_gamma to 1 after n epochs. -1 for never.
        """
        @dataclass
        class Loss:
            """Parameters related to the Loss function

            :param mse: mean squared error
            :param ssim: structural similarity index measure
            :param dice: Dice segmentation coefficient
            """
            mse: float = 0
            ssim: float = 1
            dice: float = 0

            def __post_init__(self):
                total = sum(self.__dict__.values())
                for key in self.__dict__.keys():
                    setattr(self, key, getattr(self, key) / total)

        epochs: int = 5
        folds: int = 3
        loss: Loss = field(default_factory=Loss)
        lr: float = 0.001
        lr_gamma: float = 0.95
        lr_decay_end: int = 3

    in_dir: Path
    out_dir: Path
    yaml: InitVar[str] = None
    debug: bool = False
    data: Data = field(init=False, default_factory=Data)
    model: Model = field(init=False, default_factory=Model)
    train: Train = field(init=False, default_factory=Train)

    def __post_init__(self, yaml: str):
        self.in_dir = Path(self.in_dir)
        self.out_dir = Path(self.out_dir)
        if self.debug:
            yaml = importlib.resources.read_text('reconai.resources', 'config_debug.yaml')
        elif not yaml:
            yaml = importlib.resources.read_text('reconai.resources', 'config_default.yaml')
        self._yaml = strict_load(yaml)

        args = [
            datetime.now().strftime("%Y%m%d-%H%M"),
            'CRNN-MRI' + '' if self.model.bcrnn else 'b',
            f'R{self.data.undersampling}',
            f'E{self.train.epochs}',
            'DEBUG' if self.debug else None
        ]

        __deep_update__(self, self._yaml)

        self._yaml['name'] = '_'.join(a for a in args if a)
        self._yaml['date'] = args[0]
        self._yaml['in_dir'] = self.in_dir.as_posix()
        self._yaml['out_dir'] = self.out_dir.as_posix()
        self._yaml['debug'] = self.debug

    def mkoutdir(self):
        self.out_dir = self.out_dir / self.name
        self.out_dir.mkdir(exist_ok=True, parents=True)
        with open(self.out_dir / 'config.yaml', 'w') as f:
            f.write(str(self))

    @property
    def name(self) -> str:
        return self._yaml['name'].value

    def as_dict(self):
        return __deep_dict__(self)

    def __str__(self):
        return self._yaml.as_yaml()


types = (int, float, bool, str)


def __deep_dict__(obj) -> dict:
    r = {key: value for key, value in obj.__dict__.items() if not key.startswith('_')}
    for key in r.keys():
        if is_dataclass(r[key]):
            r[key] = __deep_dict__(r[key])
    return r


def __deep_update__(obj, yaml: YAML):
    for key, value in yaml.items():
        key = key.value
        if value.is_mapping():
            __deep_update__(getattr(obj, key), value)
        else:
            ty = type(getattr(obj, key))
            for t in types:
                if ty == t:
                    setattr(obj, key, t(value.value))
                    break
