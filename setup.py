from setuptools import setup, find_packages

version = '1.10.0'

if __name__ == '__main__':
    try:
        import torch
    except ModuleNotFoundError as e:
        e.msg += '\nMake sure PyTorch is preinstalled, as it must be installed with CUDA included!'
        e.msg += '\npip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121'
        print(e)

    if not torch.backends.cuda.is_built():
        print('WARNING: No CUDA installed!')

    setup(
        name='reconai',
        version=version,
        license='MIT',
        py_modules=[],
        author='C.R. Noordman',
        author_email='stan.noordman@radboudumc.nl',
        description='',
        packages=find_packages(),
        package_data={"reconai.resources": ["*.yaml"]},
        install_requires=[
            'numpy~=1.24',
            'SimpleITK~=2.2',
            'scipy~=1.10',
            'opencv-python~=4.7',
            'scikit-image~=0.21',
            'strictyaml>=1.7',
            'click>=8.1',
            'piqa>=1.3',
            'wandb>=0.15',
            'nnunetv2~=2.2',
            'monai~=1.3'
        ],
        extras_require={
            'dev': [
                'pytest',
                'pytest-click',
                'coverage',
                'flake8'
            ],
            'fire': [
                'fire @ git+https://github.com/DIAGNijmegen/FastIMRI-FIRE.git'
            ]
        },
        entry_points={
            'console_scripts': [
                'reconai = reconai.__main__:cli',
            ],
        }
    )
