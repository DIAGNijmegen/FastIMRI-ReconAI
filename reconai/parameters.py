import importlib
from dataclasses import dataclass, field, InitVar
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
        :param q: Quality preference. (0 <= q <= 1). Higher means fewer sequences per case, with each sequence containing primarily center slices. Lower means more sequences per case, with each sequence containing primarily edge slices.
        :param normalize: image normalize divisor
        :param equal_images: keep images equal across a sequence
        :param equal_masks: keep masks equal across a sequence
        :param multislice: ?
        :param undersampling: how many k-lines to synthetically remove
        :param mask_seed: seed to Gaussian random k-space masks
        """
        split_regex: str = '.*_(.*)_'
        filter_regex: str = 'sag'
        shape_x: int = 256
        shape_y: int = 256
        sequence_length: int = 5
        sequence_seed: int = 11
        q: float = 0.5
        normalize: float = 1961.06
        equal_images: bool = True
        equal_masks: bool = True
        multislice: bool = False
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
        :param folds: number of folds. if less than 3, folds = 1
        :param kernelsize: CRNN kernel convolution size
        :param channels: CRNN channel width
        :param layers: CRNN total layers
        :param bcrnn: whether to include the BCRNN layer (False, to replace with regular CRNN layer)
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

        epochs: int = 5
        folds: int = 64
        loss: Loss = field(default_factory=Loss)
        lr: float = 0.001
        lr_gamma: float = 0.95
        stop_lr_decay: int = 45

    in_dir: Path
    out_dir: Path
    yaml: InitVar[str] = None
    debug: bool = False
    data: Data = field(init=False, default_factory=Data)
    model: Model = field(init=False, default_factory=Model)
    train: Train = field(init=False, default_factory=Train)

    def __post_init__(self, yaml: str):

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
        self._name = '_'.join(a for a in args if a)

        __deep_update__(self, self._yaml)
        self.out_dir = self.out_dir / self._name
        self.out_dir.mkdir(exist_ok=True, parents=True)
        with open(self.out_dir / 'config.yaml', 'w') as f:
            f.write(str(self))

    @property
    def name(self) -> str:
        return self._name + '_DEBUG' if self.debug else ''

    def __str__(self):
        return self._yaml.as_yaml()


types = (int, float, bool, str)


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
