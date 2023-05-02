from dataclasses import dataclass, field
from typing import Literal, Optional, TypedDict

import numpy as np
from rdkit import Chem

BACKBONE_NAMES = ["CA", "C", "N", "O"]
AA_NAME_TO_SYM = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

AA_NAME_TO_INDEX = {k: i for i, k in enumerate(AA_NAME_TO_SYM.keys())}


class PDBHeaderLine(TypedDict):
    """The parsed information of the HEADER line."""

    type: Literal["HEADER"]
    value: str


class PDBUnhandledLine(TypedDict):
    """Placeholder for the unhandled line."""

    type: Literal["UNHANDLED"]


class PDBAtomLine(TypedDict):
    """The parsed information of an atom in a single line of pdb.

    [reference document](https://www.biostat.jhsph.edu/~iruczins/teaching/260.655/links/pdbformat.pdf)
    ex) ATOM   2375  CD  PRO A 315      36.570  41.741  35.315  1.00 39.74           C
    unused info: temperature factor: 39.74
    """

    line: str  # the line itself
    type: Literal["ATOM"]
    atom_id: int  # 2375
    atom_name: str  # CD (a type of carbons)
    residue_name: str  # PRO
    chain_id: str  #  A
    residue_id: int  # 315
    residue_insert_id: str  # ""
    x: float  # 36.570
    y: float  # 41.741
    z: float  # 35.315
    occupancy: float  # 1.00
    segment_id: str  # ""
    element_symb: str  # C
    charge: str  # ""


def get_residue_signature(atom: PDBAtomLine) -> str:
    """get the stringified residue signature."""
    return "_".join(
        [
            atom["chain_id"],
            atom["segment_id"],
            str(atom["residue_id"]),
            atom["residue_insert_id"],
        ]
    )


@dataclass(slots=True)
class PDBAtomData:
    """A set of parsed atom data."""

    # atoms: list[PDBAtomLine] = field(default_factory=list)
    atomic_numbers: list[int] = field(default_factory=list)
    atomic_weight: list[float] = field(default_factory=list)
    pos: list[np.ndarray] = field(default_factory=list)  # np.array with [x,y,z] dtype=np.float32
    atom_name: list[str] = field(default_factory=list)
    is_backbone: list[bool] = field(default_factory=list)
    atom_aa_index: list[int] = field(default_factory=list)

    def add_atom(self, atom: PDBAtomLine) -> None:
        """add information of an atom from pdb file.

        Args:
            atom (PDBAtomInfo): information of an atom from a single line of the pdb file.
        """
        ptable = Chem.GetPeriodicTable()
        atomic_number = ptable.GetAtomicNumber(atom["element_symb"])
        atomic_weight = ptable.GetAtomicWeight(atomic_number)

        # self.atoms.append(atom)
        self.atomic_numbers.append(atomic_number)
        self.atomic_weight.append(atomic_weight)
        self.pos.append(np.array([atom["x"], atom["y"], atom["z"]], dtype=np.float32))
        self.atom_name.append(atom["atom_name"])
        self.is_backbone.append(atom["atom_name"] in BACKBONE_NAMES)
        self.atom_aa_index.append(AA_NAME_TO_INDEX[atom["residue_name"]])


@dataclass(slots=True, init=False)
class PDBAtomNumpy:
    """A set of numpy arrays of parsed atom data."""

    atomic_numbers: np.ndarray = field(init=False)
    pos: np.ndarray = field(init=False)  # np.array with [x,y,z] dtype=np.float32
    atom_name: np.ndarray = field(init=False)
    is_backbone: np.ndarray = field(init=False)
    aa_type: np.ndarray = field(init=False)  # amino acid type(index)

    def __init__(self, atom_data: PDBAtomData) -> None:
        self.atomic_numbers = np.array(atom_data.atomic_numbers)
        self.pos = np.array(atom_data.pos)
        self.atom_name = np.array(atom_data.atom_name)
        self.is_backbone = np.array(atom_data.is_backbone)
        self.aa_type = np.array(atom_data.atom_aa_index)


@dataclass(slots=True)
class PDBResidueNumpy:  # pylint: disable=invalid-name
    """A set of numpy arrays of parsed residue data."""

    aa_type: np.ndarray = field(init=False)  # amino acid type(index)
    center_of_mass: np.ndarray = field(init=False)  # center of mass
    pos_CA: np.ndarray = field(init=False)  # position of CA
    pos_C: np.ndarray = field(init=False)  # position of C
    pos_N: np.ndarray = field(init=False)  # position of N
    pos_O: np.ndarray = field(init=False)  # position of O

    def __init__(self, num_residues: int):
        self.aa_type = np.zeros(num_residues, dtype=np.int64)
        self.center_of_mass = np.zeros((num_residues, 3), dtype=np.float32)
        self.pos_CA = np.zeros((num_residues, 3), dtype=np.float32)
        self.pos_C = np.zeros((num_residues, 3), dtype=np.float32)
        self.pos_N = np.zeros((num_residues, 3), dtype=np.float32)
        self.pos_O = np.zeros((num_residues, 3), dtype=np.float32)
