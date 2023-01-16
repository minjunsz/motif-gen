# pylint: disable=all
from typing import Iterable

class UIntSparseIntVect:
    def __init__(self, length: int) -> None: ...
    def GetLength(self) -> int: ...
    def GetNonzeroElements(self) -> dict: ...
    def GetTotalVal(self, useAbs: bool = False) -> int: ...
    def ToBinary(self) -> bytes: ...
    def ToList(self) -> list[int]: ...
    def UpdateFromSequence(self, indices: Iterable[int]) -> None: ...
