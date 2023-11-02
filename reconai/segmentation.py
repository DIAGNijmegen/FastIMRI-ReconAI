import shutil
import os
import sys
import json
import pkg_resources
from pathlib import Path

from . import version


argv_0 = sys.argv[0]
nnUNet_dataset_id = '111'
nnUNet_dataset_name = f'Dataset{nnUNet_dataset_id}_FastIMRI'


def train(in_dir: Path, annotation_dir: Path, out_dir: Path, folds: int, debug: bool = False):
    raw, preprocessed, results = nnunet2_dirnames()

    existing = not annotation_dir and not out_dir
    if existing:
        nnunet2_verify_raw_dir(in_dir)
        out_dir = in_dir
    else:
        nnunet_prepare_data(in_dir, annotation_dir, out_dir)

    # out_dir is the parent of nnUNet_raw
    preprocess_dir = out_dir / preprocessed
    if preprocess_dir.exists() and not existing:
        shutil.rmtree(preprocess_dir)

    nnunet2_environ_set(out_dir)

    nnunet2_plan_and_preprocess(existing)
    nnunet2_train(configs := ['2d'] if debug else ['2d', '3d_fullres'],
                  folds := ['0'] if debug else [str(f) for f in range(folds)],
                  existing,
                  debug)
    nnunet2_find_best_configuration(configs, folds, debug)


def test(in_dir: Path, nnunet_dir: Path, out_dir: Path):
    nnunet2_environ_set(nnunet_dir)

    from nnunetv2.inference.predict_from_raw_data import predict_entry_point

    nnunet2_verify_results_dir(nnunet_dir)
    assert all([file.name.endswith('_0000.mha') for file in in_dir.iterdir() if file.suffix == '.mha']), (
        NameError(f'not all files in {in_dir} end with _0000.mha'))

    with open(nnunet_dir / 'nnUNet_results' / nnUNet_dataset_name / 'inference_information.json', 'r') as f:
        inference_information = json.load(f)
    assert inference_information['dataset_name_or_id'] == nnUNet_dataset_name

    selected_model = inference_information['best_model_or_ensemble']['selected_model_or_models'][0]
    config, plans, trainer = selected_model['configuration'], selected_model['plans_identifier'], selected_model['trainer']
    folds = Path(inference_information['best_model_or_ensemble']['some_plans_file']).parent.name[-1]

    sys.argv = [argv_0, '-d', nnUNet_dataset_name, '-i', in_dir.as_posix(), '-o', out_dir.as_posix(),
                '-f', *folds, '-c', config, '-tr', trainer, '-p', plans]
    predict_entry_point()


def nnunet2_dirnames() -> tuple[str, str, str]:
    return 'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'


def nnunet2_environ_set(base_dir: Path):
    """
    Environment paths are set when importing nnunet2 for the first time.
    Also, monkeypatches shutil.copy where necessary
    """
    assert 'nnunetv2' not in globals(), ImportError('nnunetv2 is already imported! environ is not yet set.')
    for name in nnunet2_dirnames():
        os.environ[name] = (base_dir / name).resolve().as_posix()

    import distutils.file_util
    distutils_file_util_copy_file = distutils.file_util.copy_file
    def copy_file_(src, dst, preserve_mode=1, preserve_times=1, update=False,
                  link=None, verbose=True, dry_run=False):
        distutils_file_util_copy_file(src, dst, preserve_mode=False, preserve_times=False, update=update, link=link, verbose=verbose, dry_run=dry_run)
    distutils.file_util.copy_file = copy_file_
    import nnunetv2.experiment_planning.experiment_planners.default_experiment_planner
    shutil.copy = shutil.copyfile


def nnunet2_plan_and_preprocess(existing: bool):
    from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry

    argv_existing = ['--verify_dataset_integrity'] if not existing else []
    sys.argv = [argv_0, '-d', nnUNet_dataset_id] + argv_existing

    print('plan and preprocessing')
    plan_and_preprocess_entry()


def nnunet2_train(configs: list[str], folds: list[str], existing: bool, debug: bool = False):
    from nnunetv2.run.run_training import run_training_entry

    argv_existing = ['--c'] if existing else []
    argv_debug = ['-tr', 'nnUNetTrainer_FastIMRI_debug'] if debug else []

    nnunet_trainer_path = Path(pkg_resources.get_distribution('nnunetv2').location) / 'nnunetv2/training/nnUNetTrainer'
    if debug:
        nnUNetTrainer_FastIMRI_debug = '''
from .nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_FastIMRI_debug(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset, device):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 3
        '''
        with open(nnunet_trainer_path / 'nnUNetTrainer_FastIMRI_debug.py', 'w') as f:
            f.write(nnUNetTrainer_FastIMRI_debug)

    for config in configs:
        for fold in folds:
            sys.argv = [argv_0, nnUNet_dataset_id, config, fold, '--npz'] + argv_existing + argv_debug

            print(f'training config {config}, fold {fold}')
            run_training_entry()


def nnunet2_find_best_configuration(configs: list[str], folds: list[str], debug: bool = False):
    from nnunetv2.evaluation.find_best_configuration import find_best_configuration_entry_point

    argv_debug = ['-tr', 'nnUNetTrainer_FastIMRI_debug'] if debug else []
    sys.argv = [argv_0, nnUNet_dataset_id, '-c', *configs, '-f', *folds] + argv_debug

    print('finding best configuration')
    find_best_configuration_entry_point()


def nnunet2_verify_results_dir(base_dir: Path):
    nnUNet_results = base_dir / 'nnUNet_results'
    dataset = nnUNet_results / nnUNet_dataset_name
    assert dataset.exists()
    assert (dataset / 'inference_information.json').exists()


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


def nnunet_prepare_data(data_dir: Path, annotations_dir: Path, out_dir: Path):
    nnUNet_raw = out_dir / 'nnUNet_raw'
    assert not nnUNet_raw.exists()

    dataset = nnUNet_raw / nnUNet_dataset_name

    assert next(data_dir.iterdir()).suffix == '.mha'
    data_dir_files, annotation_dir_files = {file.name for file in data_dir.iterdir()}, {file.name for file in
                                                                                        annotations_dir.iterdir()}
    assert annotation_dir_files.issubset(data_dir_files), data_dir_files.symmetric_difference(annotation_dir_files)

    nnunet_copy(data_dir, dataset / 'imagesTr', '_0000')
    nnunet_copy(annotations_dir, dataset / 'labelsTr')

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


def nnunet_copy(source: Path, target: Path, suffix: str = ''):
    target.mkdir(parents=True, exist_ok=False)
    for file in source.iterdir():
        shutil.copyfile(file, target / (file.stem + suffix + file.suffix))
