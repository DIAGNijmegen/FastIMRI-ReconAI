import re
from typing import List, Dict, Tuple, Iterable


class Sequence:
    def __init__(self, case: str, seq: Dict[str, List[int]]):
        self._sequence = seq
        for key in seq.keys():
            if not re.search('^\\d+_\\d+$', key):
                raise KeyError(f"{key} in {case} sequence is not of format '\\d+_\\d+'!")
        self._case = case

    @property
    def case(self) -> str:
        return self._case

    def __len__(self):
        return sum([len(s) for s in self._sequence.values()])

    def __eq__(self, other) -> bool:
        if isinstance(other, Sequence):
            return other._sequence == self._sequence
        elif isinstance(other, dict):
            return other == self._sequence
        else:
            return False

    def __repr__(self):
        return repr(self._sequence)

    def items(self) -> Iterable[Tuple[int, List[int]]]:
        for key in sorted(self._sequence.keys()):
            yield int(re.search('\\d+$', key).group()), self._sequence[key]


class SequenceCollection:
    def __init__(self, sequences: Dict[str, List[Sequence]]):
        # {case: a list of {i_mha: sequences as slice ids}}
        self._sequences = sequences

    def __len__(self):
        return len([seq for seqs in self._sequences.values() for seq in seqs])

    def __eq__(self, other: Dict[str, List[Sequence]]) -> bool:
        return other == self._sequences

    def __repr__(self):
        return repr(self._sequences)

    def items(self):
        for case in self._sequences.keys():
            for seq in self._sequences[case]:
                yield seq
