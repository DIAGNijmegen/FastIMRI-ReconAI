from pathlib import Path
import collections.abc
from dataclasses import dataclass, field, InitVar
from importlib import resources
from typing import cast
from box import Box

from yaml import dump
from strictyaml import load as strict_load, dirty_load, CommaSeparated, Str, Int, Float, Map, \
    EmptyDict, FixedSeq, YAML, Bool


class Config(Box):
    pass


data = Map({
    "split_regex": Str(),
    "filter_regex": Str(),
    "shape_y": Int(),
    "shape_x": Int(),
    "sequence_length": Int(),
    "mean_slices_per_mha": Int(),
    "max_slices_per_mha": Int(),
    "q": Float(),
    "normalize": Float(),
    "equal_images": Bool(),
    "sequence_seed": Int(),
    "expand_to_n": Bool(),
    "multislice": Bool(),
    "undersampling": Int(),
    "equal_masks": Bool(),
    "mask_seed": Int()
})


model = Map({
    "iterations": Int(),
    "filters": Int(),
    "kernelsize": Int(),
    "channels": Int(),
    "layers": Int()
})


train = Map({
    "epochs": Int(),
    "folds": Int(),
    "loss": Map({
        "mse": Float(),
        "ssim": Float(),
        "dice": Float(),
    }),
    "lr": Float(),
    "lr_gamma": Float(),
    "stop_lr_decay": Int(),
})


experiment = EmptyDict() | Map({

})

schema = Map({
    "data": data,
    "train": train,
    "model": model,
    "experiment": experiment
})


@dataclass
class Parameters:
    @dataclass
    class Data:
        """Parameters related to the dataset

        :param split_regex: name of the item
        :param filter_regex: price in USD per unit of the item
        :param shape_x: number of units currently available
        :param shape_y: number of units currently available
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
        iterations: int = 5
        filters: int = 64
        kernelsize: int = 3
        channels: int = 1
        layers: int = 5

    @dataclass
    class Train:
        @dataclass
        class Loss:
            mse: float = 0
            ssim: float = 1
            dice: float = 0

        epochs: int = 5
        folds: int = 64
        loss: Loss = field(default_factory=Loss)
        lr: float = 0.001
        lr_gamma: float = 0.95
        stop_lr_decay: int = 45

    yaml: InitVar[str] = None
    data: Data = field(init=False, default_factory=Data)
    model: Model = field(init=False, default_factory=Model)
    train: Train = field(init=False, default_factory=Train)

    def __post_init__(self, yaml: str):
        if yaml:
            __deep_update__(self, strict_load(yaml))


def __deep_update__(obj, yaml: YAML):
    for key, value in yaml.items():
        key = key.value
        if value.is_mapping():
            __deep_update__(getattr(obj, key), value)
        else:
            ty = type(getattr(obj, key))
            if ty == int:
                setattr(obj, key, int(value.value))
            elif ty == float:
                setattr(obj, key, float(value.value))
            else:
                setattr(obj, key, value.value)


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def apply_defaults(yaml_input: str):
    yaml_input = strict_load(yaml_input)
    yaml_default = strict_load(resources.read_text(f'{__package__}.resources', 'config_default.yaml'), schema)

    if yaml_input.data:
        yaml_final = deep_update(yaml_default.data, yaml_input.data)
        return dirty_load(dump(yaml_final), schema, allow_flow_style=True)
    else:
        return yaml_default


def load(path: Path) -> YAML:
    with open(path) as f:
        return apply_defaults(f.read())


def load_str(yaml: str = '') -> YAML:
    return apply_defaults(yaml)
