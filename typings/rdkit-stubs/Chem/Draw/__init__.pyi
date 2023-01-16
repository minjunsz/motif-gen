# pylint: disable=all
from typing import Iterable, Optional, Sequence

from IPython.core.display import Image
from rdkit.Chem import Mol

def MolsToGridImage(
    mols: Iterable[Mol],
    molsPerRow: int = 3,
    subImgSize: Sequence[int] = (200, 200),
    legends: Optional[Iterable[str]] = None,
    highlightAtomLists: Optional[Iterable[Iterable[int]]] = None,
    highlightBondLists: Optional[Iterable[Iterable[int]]] = None,
    useSVG: bool = False,
    returnPNG: bool = False,
    **kwargs
) -> Image: ...
