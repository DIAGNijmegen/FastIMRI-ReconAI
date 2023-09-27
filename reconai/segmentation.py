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

    existing = in_dir.name == raw and not annotation_dir and not out_dir
    if existing:
        nnunet2_validate_dir(in_dir)
        out_dir = in_dir.parent
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


def test(in_dir: Path, nnunet_dir: Path, debug: bool = False):
    pass


def nnunet2_dirnames() -> tuple[str, str, str]:
    return 'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'


def nnunet2_environ_set(out_dir: Path):
    """
    Environment paths are set when importing nnunet2 for the first time
    """
    for name in nnunet2_dirnames():
        os.environ[name] = (out_dir / name).resolve().as_posix()


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


def nnunet2_validate_dir(in_dir: Path):
    nnUNet_raw = in_dir
    dataset = nnUNet_raw / nnUNet_dataset_name

    assert nnUNet_raw.name == 'nnUNet_raw'
    assert nnUNet_raw.exists()
    assert dataset.name == nnUNet_dataset_name
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
    assert data_dir_files == annotation_dir_files, data_dir_files.symmetric_difference(annotation_dir_files)
    for source, target in [(data_dir, 'imagesTr'), (annotations_dir, 'labelsTr')]:
        target = dataset / target
        target.mkdir(parents=True, exist_ok=False)
        for file in source.iterdir():
            shutil.copy(file, target / (file.stem + ('_0000' if source == data_dir else '') + file.suffix))

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
