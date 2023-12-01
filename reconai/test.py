import json
import re
from copy import deepcopy
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
from PIL import Image

from reconai import version
from reconai.data import preprocess_as_variable, DataLoader, Dataset
from reconai.evaluation import Evaluation
from reconai.model.model_pytorch import CRNNMRI
from reconai.parameters import TestParameters
from reconai.print import print_log, print_version
from reconai.predict import predict, prediction_strategies, Prediction
from reconai.random import rng
from reconai.segmentation import nnunet2_segment, nnunet2_verify_results_dir


def test(params: TestParameters, nnunet_dir: Path, annotations_dir: Path):
    print_version(params.meta.name)

    version_re = r'(\d+)\.(\d+)\.(\d+)'
    version_r, params_version_r = re.match(version_re, version), re.match(version_re, params.meta.version)
    for v in [1, 2]:
        if int(version_v := version_r.group(v)) != int(params_version_v := params_version_r.group(v)):
            if v == 1:
                raise ImportError(f'major version mismatch: {version_v} != {params_version_v}')
            else:
                print(f'WARNING: minor version mismatch: {version_v} != {params_version_v}')

    if not torch.cuda.is_available():
        raise Exception('Can only run in Cuda')

    nnunet_out: Path | None = None
    nnunet_enabled = nnunet_dir and annotations_dir
    if nnunet_enabled:
        print_log('nnUNet enabled')
        assert nnunet_dir, 'nnUNet base directory is not set, but annotations directory is'
        assert annotations_dir, 'annotations directory is not set, but nnUNet base directory is'
        nnunet2_verify_results_dir(nnunet_dir)

        if annotations_dir.exists():
            nnunet2_images = {file.stem + file.suffix for file in params.in_dir.iterdir() if file.suffix == '.mha'}
            nnunet2_annotations = {file.name for file in annotations_dir.iterdir() if file.suffix == '.mha'}
            assert nnunet2_images == nnunet2_annotations, f'{params.in_dir} and {annotations_dir} contents do not match'
            print_log(f'{params.in_dir} and {annotations_dir} contents match')
        else:
            print(f'segmenting {params.in_dir} to {annotations_dir}...')
            nnunet2_segment(params.in_dir, nnunet_dir, annotations_dir)
            print_log('nnUNet complete; run this command again with exact same parameters to use')
            return

    dataset_test = Dataset(params.in_dir, normalize=params.data.normalize, sequence_len=params.data.sequence_length)

    network = CRNNMRI(n_ch=params.model.channels,
                      nf=params.model.filters,
                      ks=params.model.kernelsize,
                      nc=params.model.iterations,
                      nd=params.model.layers,
                      bcrnn=params.model.bcrnn
                      ).cuda()

    evaluator_volume = Evaluation(params)
    params_1 = deepcopy(params)
    params_1.data.sequence_length = 1
    evaluator_slice = Evaluation(params_1)

    network.load_state_dict(torch.load(params.npz))
    network.eval()

    print_log(f'model parameters: {sum(p.numel() for p in network.parameters() if p.requires_grad)}',
              f'data: {len(dataset_test)} items',
              f'saving results to {params.out_dir.resolve()}')

    params.mkoutdir()
    if nnunet_enabled:
        (nnunet_out := params.out_dir / 'nnunet').mkdir()

    multiple = params.data.sequence_length > 1
    rng(params.data.seed)
    torch.manual_seed(params.data.seed)
    with torch.no_grad():
        dataloader_test = DataLoader(dataset_test, batch_size=params.train.batch_size)

        for batch in dataloader_test:
            im_u, k_u, mask, gnd = preprocess_as_variable(batch['data'], params.data.undersampling)
            paths = batch['paths']
            for i in range(len(paths)):
                j = i + 1
                path = Path(paths[i])
                for e in evaluator_volume, evaluator_slice:
                    e.start_timer()
                pred, _ = network(im_u[i:j], k_u[i:j], mask[i:j], test=True)
                evaluator_volume.calculate_reconstruction(pred, gnd[i:j], key=path.stem[:-5])

                for s in range(params.data.sequence_length):
                    t = s + 1
                    pred_single = pred[..., s:t] if multiple else pred
                    gnd_single = gnd[i:j, ..., s:t] if multiple else gnd[i:j, ...]
                    key = f'{path.stem[:-5]}_{s}' if multiple else f'{path.stem[:-5]}_{batch["slice"][i]}'

                    evaluator_slice.calculate_reconstruction(pred_single, gnd_single, key=key)
                    img = Image.fromarray((pred_single.squeeze() * 255).byte().cpu().numpy())
                    img.save(params.out_dir / f'{key}.png')

                if nnunet_enabled:
                    # prepare images as mha for nnunet
                    sitk_image = sitk.GetImageFromArray(pred.squeeze(dim=(0, 1)).cpu().numpy().transpose(2, 0, 1))
                    sitk.WriteImage(sitk_image,
                                    nnunet_out / (path.name if params.data.sequence_length > 1 else f'{key}_0000.mha'))

    del network
    torch.cuda.empty_cache()

    stats = {}

    if nnunet_enabled:
        print_log('calculating DICE scores...')
        (nnunet_out_segment := nnunet_out / 'segmentations').mkdir()
        nnunet2_segment(nnunet_out, nnunet_dir, nnunet_out_segment)

        gnd_segmentations = {a.stem: a for a in annotations_dir.iterdir() if a.suffix == '.mha'}
        for stem, path_gnd in gnd_segmentations.items():
            for path_pred in nnunet_out_segment.iterdir():
                if path_pred.stem.split('_')[:3] == path_gnd.stem.split('_')[:3]:
                    gnd = sitk.GetArrayFromImage(mha := sitk.ReadImage(path_gnd.as_posix()))
                    pred = sitk.GetArrayFromImage(sitk.ReadImage(path_pred.as_posix()))

                    s = -1
                    if not multiple:
                        s = int(re.search(r'_(\d+)\.mha', path_pred.name).group(1))
                        gnd = gnd[s:s+1, ...]

                    evaluator_volume.calculate_dice(pred, gnd, key=path_pred.stem)

                    if (path_gnd_json := path_gnd.with_suffix('.json')).exists():
                        with open(path_gnd_json, 'r') as f:
                            gnd_json = json.load(f)
                        target_direction_gnd = (*gnd_json['inner'][:2], gnd_json['angle'])
                        slice_gnd = gnd_json['slice']

                        spacing = mha.GetSpacing()[:2]
                        if multiple or slice_gnd == s:
                            for strategy in prediction_strategies:
                                pred_single = pred[slice_gnd] if multiple else pred
                                key = f'{path_pred.stem}_{slice_gnd}' if multiple else path_pred.stem
                                evaluator_slice.calculate_target_direction(pred_single, target_direction_gnd,
                                                                       spacing=spacing, strategy=strategy, key=key)
                                prediction = predict(pred_single, strategy=strategy)
                                fn = f'{path_pred.stem}_{slice_gnd}_{strategy}.png' if multiple else f'{path_pred.stem}_{strategy}.png'
                                if prediction:
                                    prediction.show(*target_direction_gnd, save=params.out_dir / fn)
                                else:
                                    Prediction(pred_single, *target_direction_gnd).show(0, 0, 0, save=params.out_dir / fn)

                    for suffix, segmentation in [('gnd', gnd), ('pred', pred)]:
                        for s in range(params.data.sequence_length if multiple else 1):
                            if suffix == 'pred':
                                key = f'{path_pred.stem}_{s}' if multiple else path_pred.stem
                                evaluator_slice.calculate_dice(pred[s], gnd[s], key=key)
                            img = Image.fromarray((segmentation[s] * 255).astype(np.uint8))
                            fn = f'{path_pred.stem}_{s}_{suffix}.png' if multiple else f'{path_pred.stem}_{suffix}.png'
                            img.save(params.out_dir / fn)

    if params.data.sequence_length > 1:
        stats |= evaluator_volume.criterion_value_per_key
    stats |= evaluator_slice.criterion_value_per_key

    with open(params.out_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print_log(f'complete; test results are in {params.out_dir}')
