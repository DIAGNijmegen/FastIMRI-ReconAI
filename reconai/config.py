from pathlib import Path
import collections.abc
from importlib import resources
from box import Box

from yaml import dump
from strictyaml import load as strict_load, dirty_load, CommaSeparated, Str, Int, Float, Map, EmptyDict, FixedSeq, YAML


class Config(Box):
    pass


data = Map({
    "split_regex": Str(),
    "shape": FixedSeq([Int()] * 3) | CommaSeparated(Int()),
    "slices": Int()
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
    "undersampling": Int(),
    "seed": Int()
})

experiment = EmptyDict() | Map({

})

schema = Map({
    "data": data,
    "train": train,
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
