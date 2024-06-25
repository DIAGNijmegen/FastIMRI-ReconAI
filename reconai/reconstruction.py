import json
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
import torch.utils.data as torch_data
import wandb
import cv2

from reconai.data import preprocess_simulated, DataLoader, Dataset, image
from reconai.evaluation import Evaluation
from reconai.model.model_pytorch import CRNNMRI
from reconai.parameters import ModelTrainParameters, ModelParameters
from reconai.print import print_log, print_version
from reconai.random import rng
from reconai.math.fourier import fft2c, ifft2c


# def get_gpu_memory():
#     command = "nvidia-smi --query-gpu=memory.free --format=csv"
#     memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
#     memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
#     return memory_free_values
#
#
# def view(x: torch.Tensor):
#     plt.imshow(x.cpu(), cmap='gray')
#     plt.show()


@contextmanager
def reconstruct(params: ModelParameters, png: bool = False):
    print_version(params.meta.name)

    network = CRNNMRI(n_ch=params.model.channels,
                      nf=params.model.filters,
                      ks=params.model.kernelsize,
                      nc=params.model.iterations,
                      nd=params.model.layers,
                      bcrnn=params.model.bcrnn
                      ).cuda()

    network.load_state_dict(torch.load(params.npz))
    network.eval()

    # multiple = params.data.sequence_length > 1
    rng(params.data.seed)
    torch.manual_seed(params.data.seed)

    def func(file: Path, out: Path):
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            (tempdir / ('scan' + file.suffix)).symlink_to(file.resolve())
            with torch.no_grad():
                datapiece = DataLoader(Dataset(tempdir, params))
                for piece in datapiece:
                    if png:
                        np.save(out.with_name(f'{out.stem}_in.npy'), piece['data'].numpy())

                    im_u, k_u, mask, _ = preprocess_simulated(piece['data'].numpy(), params.data.undersampling)
                    for i in range(len(piece['paths'])):
                        j = i + 1

                        pred, _ = network(im_u[i:j], k_u[i:j], mask[i:j], test=True)
                        pred = pred.squeeze(dim=(0, 1)).cpu().numpy().transpose(2, 0, 1)

                        if png:
                            cv2.imwrite(out.with_suffix('.png').resolve().as_posix(), image(pred[-1]))
                            np.save(out.with_name(f'{out.stem}_out.npy'), pred[-1])

                        sitk_image = sitk.GetImageFromArray(pred)
                        sitk_image.SetOrigin([float(o[i]) for o in piece['origin']])
                        sitk_image.SetDirection([float(d[i]) for d in piece['direction']])
                        sitk_image.SetSpacing([float(d[i]) for d in piece['spacing']])
                        sitk.WriteImage(sitk_image, out.resolve())

    try:
        yield func
    except Exception as e:
        raise e
    finally:
        del network
        torch.cuda.empty_cache()


def train(params: ModelTrainParameters, wandb_active: bool = False, retry: bool = False):
    print_version(params.meta.name)
    print_log(str(params))

    if not torch.cuda.is_available():
        raise Exception('Can only run in Cuda')

    dataset_full = Dataset(params.in_dir, params)
    dataset_fold = torch_data.random_split(dataset_full, [1 / params.train.folds] * params.train.folds)
    network = CRNNMRI(n_ch=params.model.channels,
                      nf=params.model.filters,
                      ks=params.model.kernelsize,
                      nc=params.model.iterations,
                      nd=params.model.layers,
                      bcrnn=params.model.bcrnn
                      ).cuda()

    def train_optimizer_scheduler() -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.SequentialLR]:
        o = torch.optim.Adam(network.parameters(), lr=float(params.train.lr), betas=(0.5, 0.999))

        warmup = torch.optim.lr_scheduler.LinearLR(o, start_factor=1 / params.train.lr_warmup, total_iters=params.train.lr_warmup)
        decay = torch.optim.lr_scheduler.ExponentialLR(o, gamma=params.train.lr_gamma)
        # plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, cooldown=5, verbose=True)

        return o, torch.optim.lr_scheduler.SequentialLR(o, [warmup, decay], milestones=[params.train.lr_warmup])

    def train_fold(fold):
        optimizer, scheduler = train_optimizer_scheduler()
        rng(params.data.seed)
        torch.manual_seed(params.data.seed)

        if params.train.folds > 1:
            dataset_train = torch_data.ConcatDataset(ds for f, ds in enumerate(dataset_fold) if f != fold)
            dataset_validate = dataset_fold[fold]
        else:
            dataset_train, dataset_validate = torch_data.random_split(dataset_full, [0.8, 0.2])

        steps = 0
        epoch = 0
        validate_loss_best = np.inf
        last_5_loss = []
        while epoch < params.train.epochs:
            epoch_start = datetime.now()

            evaluator_train = Evaluation(params, loss_only=True)
            evaluator_validate = Evaluation(params)

            steps_end = steps + params.train.steps
            steps_excess = steps_end % len(dataset_train) if steps_end > len(dataset_train) else 0
            indices = list(range(steps, steps_end - steps_excess)) + list(range(0, steps_excess))
            steps = steps_excess if steps_excess > 0 else steps_end

            network.train()
            for batch in DataLoader(dataset_train, batch_size=params.train.batch_size, indices=indices):
                im_u, k_u, mask, gnd = preprocess_simulated(batch['data'].numpy(), params.data.undersampling)
                for i in range(len(batch['paths'])):
                    j = i + 1
                    optimizer.zero_grad()
                    pred, _ = network(im_u[i:j], k_u[i:j], mask[i:j])
                    evaluator_train.calculate_reconstruction(pred, gnd[i:j])
                    evaluator_train.loss.backward()

                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1)
                    optimizer.step()

            network.eval()
            with torch.no_grad():
                for batch in DataLoader(dataset_validate, batch_size=params.train.batch_size, indices=params.train.steps):
                    im_u, k_u, mask, gnd = preprocess_simulated(batch['data'].numpy(), params.data.undersampling)
                    for i in range(len(batch['paths'])):
                        j = i + 1
                        evaluator_validate.start_timer()
                        pred, _ = network(im_u[i:j], k_u[i:j], mask[i:j], test=True)
                        evaluator_validate.calculate_reconstruction(pred, gnd[i:j])

            model = network.state_dict()
            stats = {'fold': fold,
                     'epoch': epoch,
                     'epoch_time': (datetime.now() - epoch_start).total_seconds(),
                     'loss_train': evaluator_train.criterion_stats('loss'),
                     'loss_validate': evaluator_validate.criterion_stats('loss'),
                     'ssim_validate': evaluator_validate.criterion_stats('ssim'),
                     'mse_validate': evaluator_validate.criterion_stats('mse'),
                     'time_validate': evaluator_validate.criterion_stats('time')
                     }
            for key in [k for k, v in stats.items() if isinstance(v, tuple)]:
                value = stats.pop(key)
                stats |= {f'{key}_min': value[0], f'{key}_mean': value[1], f'{key}_max': value[2]}

            print_log(json.dumps(stats, indent=2))
            train_loss_min, _, _ = evaluator_train.criterion_stats('loss')
            _, validate_loss, _ = evaluator_validate.criterion_stats('loss')

            if len(last_5_loss) < 5:
                last_5_loss.append(train_loss_min)
            else:
                last_5_loss = last_5_loss[1:5] + [train_loss_min]

            if len(last_5_loss) == 5 and np.polyfit(range(5), last_5_loss, 1)[0] > 0 and np.mean(last_5_loss) > 0.66:
                params.data.seed += 1
                print_log(f'exploded loss: {validate_loss}')
                if retry:
                    print_log(f'retrying fold {fold} with seed {params.data.seed}')
                    return False
                return True

            if wandb_active:
                wandb.log(stats)

            def save_model(path: Path, model_stats: dict):
                torch.save(model, path)
                with open(path.with_suffix('.json'), 'w') as f:
                    json.dump(model_stats, f, indent=4)

            save_model(params.out_dir / f'reconai_{params.model_name}_{fold}.npz', stats)

            if validate_loss <= validate_loss_best:
                validate_loss_best = validate_loss
                save_model(params.out_dir / f'reconai_{params.model_name}_{fold}_best.npz', stats)

            scheduler.step()
            epoch += 1
        return True

    print_log(f'trainable parameters: {sum(p.numel() for p in network.parameters() if p.requires_grad)}',
              f'data: {len(dataset_full)} items',
              f'saving model data to {params.out_dir.resolve()}')
    params.mkoutdir()

    start = datetime.now()
    print_log(f'starting {params.train.folds}-fold training at {start}')

    for _ in range(params.train.folds):
        while not train_fold(_):
            pass
        params.data.seed += 1

    end = datetime.now()
    print_log(f'completed training in {(end - start).total_seconds()} seconds, at {end}')
