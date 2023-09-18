from setup import version
from reconai import version as module_version


def test_version():
    assert version == module_version
