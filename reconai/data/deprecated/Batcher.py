import logging

import numpy as np
from typing import List

from .Volume import Volume


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
