import os
import subprocess
from pathlib import Path

# noinspection PyUnresolvedReferences
from .nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_debug(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset, device):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 3

    @staticmethod
    def sync():
        if nnUNet_sync := os.environ.get('nnUNet_sync'):
            Path(nnUNet_sync).mkdir(parents=True)
            if os.name == 'nt':
                subprocess.run(['robocopy', os.environ.get('nnUNet_base'), nnUNet_sync, '/E', '/SL', '/XD', nnUNet_sync])
            else:
                subprocess.run(["rsync", "-rl", os.environ.get('nnUNet_base'), nnUNet_sync])

    def on_epoch_end(self):
        super().on_epoch_end()
        self.sync()

    def on_train_end(self):
        super().on_train_end()
        self.sync()
