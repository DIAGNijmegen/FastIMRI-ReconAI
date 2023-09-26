from numpy.random import default_rng, Generator

_rng = None


def rng(seed: int = None) -> Generator:
    global _rng
    if seed is not None:
        _rng = default_rng(seed)

    if _rng is None:
        raise RuntimeError('rng not yet instantiated with a seed')

    return _rng
