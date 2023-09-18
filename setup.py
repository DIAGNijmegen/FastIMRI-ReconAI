from setuptools import setup

version = "1.1.0"

if __name__ == '__main__':
    name = 'reconai'
    __version__ = ''
    exec(open(f'{name}/__version__.py').read())

    try:
        import torch
    except ModuleNotFoundError as e:
        e.msg += '\nMake sure PyTorch and CUDA are installed!'
        raise e
    if not torch.has_cuda:
        raise ModuleNotFoundError('Make sure CUDA is installed!')

    setup(
        name=name,
        version=__version__,
        license='MIT',
        py_modules=[],
        author='C.R. Noordman',
        author_email='stan.noordman@radboudumc.nl',
        description='',
        install_requires=[
            'pytest~=7.2.2',
            'numpy~=1.24.0',
            'reconai~=0.1',
            'torch~=2.0.1',
            'matplotlib~=3.6.2',
            'SimpleITK~=2.2.1',
            'scipy~=1.10.1',
            'pandas~=1.5.3',
            'opencv-python~=4.7.0.72',
            'scikit-learn~=1.2.2',
            'scikit-image~=0.21.0',
            'PyYAML~=6.0',
            'strictyaml~=1.7.3',
            'click~=8.1.3',
            'setuptools~=60.2.0',
            'piqa~=1.3.0',
            'wandb~=0.15.9',

        ],
        extras_require={
            'dev': [
                'pytest',
                'pytest-click',
                'coverage',
                'flake8',
                'freezegun'
            ]
        },
        entry_points={
            'console_scripts': [
                'reconai = reconai.__main__:cli',
            ],
        }
    )
