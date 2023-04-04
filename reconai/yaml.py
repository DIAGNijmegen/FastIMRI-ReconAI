from pathlib import Path
import collections.abc

from yaml import dump
from strictyaml import load as strict_load, dirty_load, CommaSeparated, Str, Int, Float, Map, EmptyDict, FixedSeq


volume = Map({
    "key": Str(),
    "shape": FixedSeq([Int()] * 3) | CommaSeparated(Int()),
    "slices": Int()
})

train = Map({
    "epochs": Int(),
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
    "volume": volume,
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
    yaml_default = strict_load("""
    volume:
      key: sag
      # x, y, sequence length
      shape: 256,256,15
      # files * slices <= sequence length, else the case is skipped
      slices: 3
    train:
      epochs: 75
      loss:
        mse: 1
        ssim: 0
        dice: 0
      lr: 0.001
      undersampling: 8
      seed: -1
    experiment:
    """, schema)

    if yaml_input.data:
        yaml_final = deep_update(yaml_default.data, yaml_input.data)
        return dirty_load(dump(yaml_final), schema, allow_flow_style=True)
    else:
        return yaml_default

def load(path: Path):
    with open(path) as f:
        return apply_defaults(f.read())


def load_str(yaml: str):
    return apply_defaults(yaml)