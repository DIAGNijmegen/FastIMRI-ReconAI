from setup import version as setup_version
from reconai import version as module_version


def test_version():
    assert setup_version == module_version
