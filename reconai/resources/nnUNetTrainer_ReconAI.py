import os
import subprocess
from pathlib import Path

# noinspection PyUnresolvedReferences
from .nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_ReconAI(nnUNetTrainer):
    @staticmethod
    def sync():
        if nnUNet_sync := os.environ.get('nnUNet_sync'):
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
