from setuptools import setup

from reconai.__version__ import __version__


if __name__ == '__main__':
    try:
        import torch
    except ModuleNotFoundError as e:
        e.msg += '\nMake sure PyTorch and CUDA are installed!'
        raise e
    if not torch.has_cuda:
        raise ModuleNotFoundError('Make sure CUDA is installed!')

    setup(
        name='fastimri_reconai',
        version=__version__,
        license='MIT',
        author='C.R. Noordman',
        author_email='stan.noordman@radboudumc.nl',
        description='',
        install_requires=[
            'numpy~=1.24',
            'SimpleITK~=2.1',
            'click',
            'matplotlib',
            'python-box~=6.0'
        ]
    )
