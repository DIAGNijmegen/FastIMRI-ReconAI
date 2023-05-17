from pathlib import Path
import collections.abc
from importlib import resources
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
    "undersampling": Int(),
    "equal_masks": Bool(),
    "mask_seed": Int()
})

experiment = EmptyDict() | Map({

})

schema = Map({
    "data": data,
    "train": train,
    "model": model,
    "experiment": experiment
})


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
