from pathlib import Path
from collections import OrderedDict

from strictyaml import load as yamlload, CommaSeparated, Map, Str, Int, Float, Map, Optional, YAMLError

volume = Map({
    Optional("key", "sag"): Str(),
    Optional("shape", [256, 256, 15]): CommaSeparated(Int()),
    Optional("slices", 3): Int()}
)
train = Map({
    Optional("epochs", 100): Int(),
    "loss": Map({
        Optional("mse", 1): Float(),
        Optional("ssim"): Float(),
        Optional("dice"): Float(),
    }),
    Optional("lr", 0.001): Float(),
    Optional("undersampling", 8): Int(),
    Optional("seed", -1): Int()})
experiment = Map({

})
schema = Map({"volume": volume, "train": train, "experiment": experiment})

def OptionalRoot():
    pass

def load(path: Path | str):
    with open(path) as f:
        t = yamlload(f.read())
        yaml = yamlload(f.read(), schema)



    pass
