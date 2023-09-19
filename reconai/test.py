from reconai import version
from reconai.parameters import TestParameters
from reconai.print import print_log


def test(params: TestParameters):
    print_log(f'reconai version {version}', params.meta.name)
    assert version == params.meta.version
