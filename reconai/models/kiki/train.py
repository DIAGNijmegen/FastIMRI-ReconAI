import datetime
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
from box import Box
from typing import List
from pathlib import Path
import logging
from os.path import join

from reconai.config import Config
from reconai.parameters import Parameters
from reconai.data.data import get_dataset_batchers, prepare_input, prepare_input_as_variable,\
    from_tensor_format, append_to_file
from reconai.utils.graph import print_acceleration_train_loss, print_acceleration_validation_loss, print_loss_progress,\
    print_prediction_error, print_full_prediction_sequence, print_loss_comparison_graphs, print_iterations
from reconai.utils.metric import complex_psnr
from reconai.models.bcrnn.model_pytorch import CRNNMRI
from reconai.models.bcrnn.module import Module
from reconai.models.kiki.models import KIKI


def train(params: Parameters):
    num_epoch = 3 if params.debug else params.config.train.epochs
    n_folds = params.config.train.folds if params.config.train.folds > 2 else 1

    # Configure directory info
    # save_dir: Path = params.out_dir / params.name_date
    # save_dir.mkdir(parents=True)
    # logging.info(f"saving model to {save_dir.absolute()}")

    # Specify network
    # {iters, k, in_ch, out_ch, fm, i}
    network = KIKI(iters=3, k=5, in_ch=256, out_ch=256, fm=64, i=5).cuda()
    optimizer = optim.Adam(network.parameters(), lr=float(params.config.train.lr),
                           betas=(0.5, 0.999), weight_decay=10**-7)
    criterion = torch.nn.MSELoss().cuda()

    train_val_batcher, test_batcher = get_dataset_batchers(params.in_dir, params.config.data.slices)

    for epoch in range(num_epoch):
        t_start = time.time()

        # train, validate, test = get_dataset_batchers(args, data, n_folds, fold)
        train_err, train_batches = 0, 0
        for im in train_val_batcher.items_fold(0, 5, validation=False):
            logging.debug(f"batch {train_batches}")
            im_u, k_u, mask, gnd = prepare_input_as_variable(im, params.config.train.undersampling)

            optimizer.zero_grad(set_to_none=True)
            rec = network(k_u[:, :, :, :, 0], mask[:, :, :, :, 0])
            loss = criterion(rec, gnd[:, :, :, :, 0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5)
            optimizer.step()

            train_err += loss.item()
            train_batches += 1

            if params.debug and train_batches == 2:
                break
        logging.info(f"completed {train_batches} train batches")