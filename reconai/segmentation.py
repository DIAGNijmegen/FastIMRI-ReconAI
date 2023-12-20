import shutil
import os
import json
import subprocess
import re
from pathlib import Path
from importlib.metadata import distribution
from importlib.resources import files

from . import version
from .print import print_version


nnUNet_dirnames = ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')
nnUNet_dataset_id = '111'
nnUNet_dataset_name = f'Dataset{nnUNet_dataset_id}_FastIMRI'
nnUNet_environ = dict(os.environ.copy())


def train(in_dir: Path, annotation_dir: Path, out_dir: Path, sync_dir: Path, folds: int, gpus: int, debug: bool = False):
    print_version()
    print(f'syncing {out_dir} --> {sync_dir}')

    existing = not annotation_dir and not out_dir
    if existing:
        nnunet2_verify_raw_dir(in_dir)
        out_dir = in_dir
    else:
        nnunet2_prepare_data(in_dir, annotation_dir, out_dir)
    nnUNet_base, nnUNet_sync = (d.resolve().as_posix() for d in (out_dir, sync_dir))

    # out_dir is the parent of nnUNet_raw
    preprocess_dir = out_dir / nnUNet_dirnames[1]
    if preprocess_dir.exists() and not existing:
        shutil.rmtree(preprocess_dir)

    with open(f'{nnUNet_base}/locals.json', 'w') as f:
        json.dump({key: str(value) for key, value in locals().items() if not key.startswith('_')}, f)

    nnunet2_prepare_nnunet(out_dir, sync_dir)
    nnunet2_plan_and_preprocess(existing)
    nnunet2_train(configs := ['2d'],# if debug else ['2d', '3d_fullres'],
                  folds := ['0'] if debug else [str(f) for f in range(folds)],
                  gpus, existing,
                  debug)
    nnunet2_find_best_configuration(configs, folds, debug)
    if sync_dir:
        if os.name == 'nt':
            subprocess.run(['robocopy', nnUNet_base, nnUNet_sync, '/E', '/SL', '/XD', nnUNet_sync])
        else:
            subprocess.run(["rsync", "-rl", nnUNet_base, nnUNet_sync])


def nnunet2_segment(in_dir: Path, nnunet_dir: Path, out_dir: Path):
    nnunet2_prepare_nnunet(nnunet_dir)
    nnunet2_verify_results_dir(nnunet_dir)
    assert all([file.name.endswith('_0000.mha') for file in in_dir.iterdir() if file.suffix == '.mha']), (
        NameError(f'not all files in {in_dir} end with _0000.mha'))

    with open(nnunet_dir / 'nnUNet_results' / nnUNet_dataset_name / 'inference_information.json', 'r') as f:
        inference_information = json.load(f)
    assert inference_information['dataset_name_or_id'] == nnUNet_dataset_name

    selected_model = inference_information['best_model_or_ensemble']['selected_model_or_models'][0]
    config, plans, trainer = selected_model['configuration'], selected_model['plans_identifier'], selected_model['trainer']
    folds = [str(f) for f in inference_information['folds']]

    args = ['-d', nnUNet_dataset_name, '-i', in_dir.as_posix(), '-o', out_dir.as_posix(),
            '-f', *folds, '-c', config, '-tr', trainer, '-p', plans]
    nnunet2_command('nnUNetv2_predict', *args)


def nnunet2_dirnames() -> tuple[str, str, str]:
    return 'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'


def nnunet2_prepare_nnunet(base_dir: Path, sync_dir: Path = None):
    nnUNet_environ['nnUNet_base'] = base_dir.resolve().as_posix()
    for name in nnUNet_dirnames:
        nnUNet_environ[name] = (base_dir / name).resolve().as_posix()
    if sync_dir:
        nnUNet_environ['nnUNet_sync'] = sync_dir.resolve().as_posix()

    nnunetv2_dir = distribution('nnunetv2').locate_file('nnunetv2/training/nnUNetTrainer')
    for resource in files('reconai.resources').iterdir():
        if resource.name.startswith('nnUNetTrainer'):
            with resource.open('r') as src:
                with open(Path(nnunetv2_dir) / resource.name, 'w') as dst:
                    dst.write(src.read())


def nnunet2_plan_and_preprocess(existing: bool):
    args_existing = ['--verify_dataset_integrity'] if not existing else []
    args = ['-d', nnUNet_dataset_id] + args_existing

    print('plan and preprocessing')
    nnunet2_command('nnUNetv2_plan_and_preprocess', *args)


def nnunet2_train(configs: list[str], folds: list[str], gpus: int, existing: bool, debug: bool = False):
    args_existing = ['--c'] if existing else []
    args_trainer = ['-tr', 'nnUNetTrainer_debug' if debug else 'nnUNetTrainer_ReconAI']

    for config in configs:
        for fold in folds:
            args = [nnUNet_dataset_id, config, fold, '--npz', '-num_gpus', str(gpus)] + args_existing + args_trainer
            print(f'training config {config}, fold {fold}')
            nnunet2_command('nnUNetv2_train', *args)


def nnunet2_verify_results_dir(base_dir: Path, debug: bool = False):
    nnUNet_results = base_dir / 'nnUNet_results'
    dataset_dir = nnUNet_results / nnUNet_dataset_name
    assert dataset_dir.exists(), \
        f'{dataset_dir} does not exist? run\nreconai test_find_configuration --nnunet_dir {base_dir.as_posix()}'


def nnunet2_find_best_configuration(configs: list[str], folds: list[str], debug: bool = False):
    args_trainer = ['-tr', 'nnUNetTrainer_debug' if debug else 'nnUNetTrainer_ReconAI']

    args = [nnUNet_dataset_id, '-c', *configs, '-f', *folds] + args_trainer
    print('finding best configuration')
    nnunet2_command('nnUNetv2_find_best_configuration', *args)


def nnunet2_verify_raw_dir(base_dir: Path):
    nnUNet_raw = base_dir / 'nnUNet_raw'
    dataset = nnUNet_raw / nnUNet_dataset_name

    assert nnUNet_raw.exists()
    assert dataset.exists()

    assert (dataset_json := dataset / 'dataset.json').exists()
    with open(dataset_json, 'r') as j:
        dataset_content = json.load(j)
    dataset_expected = ['channel_names', 'labels', 'numTraining', 'file_ending', 'overwrite_image_reader_writer']
    assert all(key in dataset_content.keys() for key in dataset_expected)

    assert (imagesTr := dataset / 'imagesTr').exists()
    assert (labelsTr := dataset / 'labelsTr').exists()
    images = set()
    for file in imagesTr.iterdir():
        assert file.stem.endswith('_0000')
        assert file.suffix == '.mha'
        images.add(file.name[:-9])
    labels = set()
    for file in labelsTr.iterdir():
        assert file.suffix == '.mha'
        labels.add(file.name[:-4])

    assert images == labels, images.symmetric_difference(labels)
    assert dataset_content['numTraining'] == len(images)


def nnunet2_prepare_data(data_dir: Path, annotations_dir: Path, out_dir: Path):
    nnUNet_raw = out_dir / 'nnUNet_raw'
    assert not nnUNet_raw.exists()

    dataset = nnUNet_raw / nnUNet_dataset_name

    data_dir_files, annotation_dir_files = (
        {file.name for file in data_dir.iterdir() if file.suffix == '.mha'},
        {file.name for file in annotations_dir.iterdir() if file.suffix == '.mha'})
    assert annotation_dir_files.issubset(data_dir_files), data_dir_files.symmetric_difference(annotation_dir_files)

    nnunet2_copy(data_dir, dataset / 'imagesTr', '_0000')
    nnunet2_copy(annotations_dir, dataset / 'labelsTr')

    with open(dataset / 'dataset.json', 'w') as j:
        json.dump({
            "channel_names": {"0": "IMRI", },
            "labels": {"background": 0, "needle": 1},
            "numTraining": len(list(data_dir.iterdir())),
            "file_ending": ".mha",
            "overwrite_image_reader_writer": "SimpleITKIO",
            "dataset_name": "imri_xyt",
            "release": version
        }, j, indent=4)


def nnunet2_copy(source: Path, target: Path, suffix: str = ''):
    target.mkdir(parents=True, exist_ok=False)
    for file in source.iterdir():
        shutil.copyfile(file, target / (file.stem + suffix + file.suffix))


def nnunet2_command(cmd: str, *args):
    print([cmd] + list(args))
    process = subprocess.run([cmd] + list(args), env=nnUNet_environ)
    if process.returncode != 0:
        raise RuntimeError(f'An error occurred while running the subprocess.\n{process.stderr}\n{process.stdout}')
    else:
        print(process.stdout)
