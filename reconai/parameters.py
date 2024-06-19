import json
import shutil
from dataclasses import dataclass, field, InitVar, is_dataclass
from datetime import datetime
from pathlib import Path

from strictyaml import load as yaml_load, YAML

from reconai import version
from .resources import config_debug


def now():
    return datetime.now().strftime("%Y%m%dT%H%M")


@dataclass
class Parameters:
    @dataclass
    class Data:
        """Parameters related to the dataset

        :param shape_x: image columns to crop or zero-fill to
        :param shape_y: image rows to crop or zero-fill to
        :param sequence_length: length of T
        :param normalize: image normalize divisor
        :param undersampling: how many k-lines to synthetically remove
        :param seed: seed to Gaussian random k-space masks
        """
        shape_x: int = 256
        shape_y: int = 256
        sequence_length: int = 5
        normalize: float = 0
        undersampling: int = 8
        seed: int = 11

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
        :param steps: number of training steps per epoch (that is, number of images to forward/backward)
        :param batch_size: batch size
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
        steps: int = 0
        batch_size: int = 10
        loss: Loss = field(default_factory=Loss)
        lr: float = 0.001
        lr_gamma: float = 0.95
        lr_warmup: int = 0
        augment_mult: int = 1
        augment_all: bool = False

    @dataclass
    class Meta:
        name: str = 'untitled'
        date: str = 'date'
        in_dir: str = 'in_dir'
        out_dir: str = 'out_dir'
        debug: bool = False
        version: str = '0.0.0'

    data: Data = field(init=False, default_factory=Data)
    model: Model = field(init=False, default_factory=Model)
    train: Train = field(init=False, default_factory=Train)
    meta: Meta = field(init=False, default_factory=Meta)

    def __post_init__(self):
        self._yaml = None

    def _load_yaml(self, yaml: str):
        self._yaml = yaml_load(yaml)
        self.meta.name = self._yaml.data.get('name', self.meta.name)
        __deep_update__(self, self._yaml)

    def mkoutdir(self):
        raise NotImplementedError()

    @property
    def name(self) -> str:
        debug = '_DEBUG' if self.meta.debug else ''
        return f'{self.meta.name}_R{self.data.undersampling}{debug}'

    @property
    def in_dir(self) -> Path:
        return Path(self.meta.in_dir)

    @property
    def out_dir(self) -> Path:
        return Path(self.meta.out_dir)

    def as_dict(self):
        return __deep_dict__(self)

    def __str__(self):
        return YAML(self.as_dict()).lines()


@dataclass
class ModelTrainParameters(Parameters):
    in_dir_: InitVar[Path] = None
    out_dir_: InitVar[Path] = None
    yaml_file: InitVar[Path | str] = None

    def __post_init__(self, in_dir_: Path, out_dir_: Path, yaml_file: Path | str):
        super().__post_init__()

        debug = False
        if isinstance(yaml_file, str):
            yaml = yaml_file
        elif not yaml_file:
            yaml = config_debug
            debug = True
        else:
            with open(yaml_file, 'r') as f:
                yaml = f.read()
        self._load_yaml(yaml)

        self.meta.name = self.name
        self.meta.date = now()
        self.meta.in_dir = Path(in_dir_).as_posix()
        self.meta.out_dir = (Path(out_dir_) / self.meta.name).as_posix()
        self.meta.debug = debug
        self.meta.version = version

    def mkoutdir(self):
        self.out_dir.mkdir(exist_ok=False, parents=True)
        with open(self.out_dir / f'config_{self.name}.yaml', 'w') as f:
            f.write(str(self))


@dataclass
class ModelParameters(Parameters):
    in_dir_: InitVar[Path] = None
    model_dir: InitVar[Path] = None
    model_name: InitVar[str] = None
    tag: InitVar[str] = None

    def __post_init__(self, in_dir_: Path, model_dir: Path, model_name: str, tag: str):
        super().__post_init__()

        if model_name:
            model = (model_dir / model_name).with_suffix('.npz')
            if model.exists():
                self._model = model
            else:
                raise FileNotFoundError(f'no model named {model_name}')
        else:
            losses: dict[Path, float] = {}
            for file in model_dir.iterdir():
                if file.suffix == '.json':
                    with open(file, 'r') as f:
                        stats = json.load(f)
                        losses[file] = stats['loss_validate_mean']
            if not losses:
                raise FileNotFoundError(f'no models found in {model_dir}')
            self._model = min(losses, key=lambda k: losses[k])

        with open(model_dir / 'config.yaml', 'r') as f:
            yaml = f.read()
        self._load_yaml(yaml)

        self.meta.in_dir = Path(in_dir_).as_posix()
        test_name = '_'.join([now(), self._model.stem] + ([tag] if tag else []))
        self.meta.out_dir = (model_dir / test_name).as_posix()

    def mkoutdir(self):
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True)

    @property
    def npz(self) -> Path:
        """
        Trained model (npz file)
        """
        return self._model.with_suffix('.npz')


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
                    try:
                        setattr(obj, key, t(value.value))
                    except AttributeError:
                        pass
                    break
