import logging, re, os
from pathlib import Path

import numpy as np
from typing import List

from .Volume import Volume


class Batcher1:
    def __init__(self, data_dir: Path, n_folds: int = 1):
        self._data_dir = data_dir
        self._n_folds = n_folds
        self._paths = []

    def load(self, regex: str = None, n: int = 0):
        for root, dirs, files in os.walk(self._data_dir):
            for file in files:
                search = re.search(regex, file) if regex else True
                if search and file.endswith('.mha'):
                    self._paths.append((root / Path(file)).relative_to(self._data_dir))
                    if 0 < n == len(self._paths):
                        return


        # for patient_dir in data_dir.iterdir():
        #     try:
        #         if patient_dir.is_dir():
        #             files = list(patient_dir.iterdir())
        #             study_ids = {fn.name.split('_')[1] for fn in files if not fn.name.startswith('tmp')}
        #             for study_id in study_ids:
        #                 data.append(Volume(study_id, [fn for fn in files if study_id in fn.name]))
        #     except:
        #         continue


class Batcher:
    batch_size: int = 1
    shuffle: bool = True

    def __init__(self, volumes: List[Volume]):
        self.volumes: List[Volume | np.ndarray] = list(volumes)
        self._blacklist = []

    def get_blacklist(self):
        for _ in self.generate():
            pass
        return self._blacklist

    def generate(self) -> np.ndarray:
        minibatch = []
        all_array = False  # self.volumes only has array elements
        while not (all_array and len(self.volumes) < self.batch_size - len(minibatch)):
            nv = len(self.volumes)

            # convert all self.volumes to arrays when it is small, while loop conditional takes effect
            if not all_array and nv <= self.batch_size:
                for item in self.volumes:
                    if isinstance(item, Volume):
                        self.volumes.remove(item)
                        try:
                            data = item.to_ndarray()
                        except Exception as e:
                            logging.warning(str(e))
                            self._blacklist.append(item.study_id)
                            continue
                        else:
                            self.volumes.extend([data[d] for d in reversed(range(len(data)))])
                all_array = True

            # select a random from volume
            nex = np.random.choice(range(nv)) if Batcher.shuffle else nv - 1
            try:
                item = self.volumes.pop(nex)
            except IndexError as e:
                logging.warning(str(e))
                break

            if isinstance(item, Volume):
                # select random array if volume splits into multiple, rest is put back
                try:
                    data = item.to_ndarray()
                except Exception as e:
                    logging.warning(str(e))
                    self._blacklist.append(item.study_id)
                    continue
                else:
                    nex = np.random.choice(range(len(data))) if Batcher.shuffle else 0
                    for d in reversed(range(len(data))):
                        if d == nex:
                            minibatch.append(data[d])
                        else:
                            self.volumes.append(data[d])
            else:
                data = item
                minibatch.append(data)

            if len(minibatch) == Batcher.batch_size:
                yield np.stack(minibatch)
                minibatch = []
