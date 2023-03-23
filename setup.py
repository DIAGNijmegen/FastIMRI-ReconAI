from setuptools import setup


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
            'numpy~=1.24',
            'SimpleITK~=2.1',
            'click',
            'matplotlib',
            'python-box~=6.0',
            'torch',
            'scikit-learn',
            'opencv-python'
        ]
    )
