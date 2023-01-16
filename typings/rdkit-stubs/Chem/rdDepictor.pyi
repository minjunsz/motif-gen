# pylint: disable=all
from typing import Iterable, Optional

from rdkit.Chem import Mol
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect

def Compute2DCoords(
    mol: Mol,
    canonOrient: bool = True,
    clearConfs: bool = True,
    coordMap: dict = {},
    nFlipsPerSample: int = 0,
    nSample: int = 0,
    sampleSeed: int = 0,
    permuteDeg4Nodes: bool = False,
    bondLength: float = -1.0,
    forceRDKit: bool = False,
) -> int: ...
def GenerateDepictionMatching2DStructure(
    mol: Mol,
    reference: Mol,
    confId: int = -1,
    acceptFailure: bool = False,
    forceRDKit: bool = False,
    allowRGroups: bool = False,
) -> tuple[tuple[int, int], ...]: ...
def GetHashedMorganFingerprint(
    mol: Mol,
    radius: int,
    nBits: int = 2048,
    invariants: Iterable[int] = [],
    fromAtoms: Iterable[int] = [],
    useChirality: bool = False,
    useBondTypes: bool = True,
    useFeatures: bool = False,
    bitInfo: Optional[dict] = None,
    includeRedundantEnvironments: bool = False,
) -> UIntSparseIntVect: ...
