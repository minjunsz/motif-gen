"""Module for parsing ligand data with '.sdf' format."""
import typing
from pathlib import Path
from typing import Union

import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType

# Valid atom families for SMARTS-based features
_AtomFamily = typing.Literal[
    "Acceptor",
    "Donor",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
    "NegIonizable",
    "PosIonizable",
    "ZnBinder",
]
ATOM_FAMILIES: list[_AtomFamily] = list(typing.get_args(_AtomFamily))
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_SDF_TO_RDKIT: dict[int, BondType] = {
    1: BondType.SINGLE,
    2: BondType.DOUBLE,
    3: BondType.TRIPLE,
    4: BondType.AROMATIC,
}


class LigandParser:
    """Parser for ligand data with '.sdf' format."""

    # Ligand as a graph G(N,E)
    num_atoms: int  # N
    num_bonds: int  # E
    feat_mat: torch.Tensor  # Size([N,_AtomFamily]) = Size([N, 8])
    positions: torch.Tensor  # Size([N,3])
    elements: torch.Tensor  # Size([N])
    center_of_mass: torch.Tensor  # Size([3])
    bond_index: torch.Tensor  # Size([2, 2E]); must be 2E to make it undirected.
    bond_type: torch.Tensor  # Size(2E); must be 2E to make it undirected.

    def __init__(self) -> None:
        fdef_name = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
        self.factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))
        self.ptable = Chem.GetPeriodicTable()

    def _get_SMARTS_feature_matrix(self, path: Union[str, Path]) -> None:  # pylint: disable=invalid-name
        """Generate a SMARTS-based feature matrix from the given ".sdf" file using rdkit.
        The sdf file is assumed to contain a single ligand molecule.
        Corresponding document for the feature extraction is [here](https://www.rdkit.org/docs/GettingStartedInPython.html#chemical-features)
        Save feature matrix as an instance variable. Matrix shape [#atoms, _AtomFamily]

        Args:
            path (Union[str, Path]): Path to the ligand file in sdf format.
        """
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = next(supplier)
        assert self.num_atoms == mol.GetNumAtoms(), "Mismatch in the number of atoms between RDKit and SDF."
        self.feat_mat = torch.zeros((self.num_atoms, len(ATOM_FAMILIES)), dtype=torch.long)
        # Extracting SMARTS-based features.
        for feature in self.factory.GetFeaturesForMol(mol):
            self.feat_mat[feature.GetAtomIds(), ATOM_FAMILIES_ID[feature.GetFamily()]] = 1

    def _parse_atoms(self, atom_block: list[str]) -> None:
        """Parse atom block in the sdf file.
        Save nodes type(`self.elements`), coordinates(`self.positions`), and the center of mass(`self.center_of_mass`)
        as an instance variable.

        Args:
            atom_block (list[str]): atom information block, parsed with `splitlines()`. Each string represents an atom.
        """
        self.positions = torch.empty((self.num_atoms, 3), dtype=torch.float)
        self.elements = torch.empty(self.num_atoms, dtype=torch.int)
        weights = torch.empty_like(self.elements)

        for idx, atom_info in enumerate(atom_block):
            info = atom_info.split()
            # parse coordinate
            x, y, z = map(float, info[:3])
            self.positions[idx, :] = torch.tensor([x, y, z])
            # parse element symbol
            symbol = info[3].capitalize()
            atomic_number = self.ptable.GetAtomicNumber(symbol)
            atomic_weight = self.ptable.GetAtomicWeight(symbol)
            self.elements[idx] = atomic_number
            weights[idx] = atomic_weight

        # calculate center of mass
        weights.unsqueeze_(dim=1)  # reshape [N] -> [N,1]
        self.center_of_mass = (self.positions * weights).sum(dim=0) / weights.sum()

    def _parse_bonds(self, bond_block: list[str]) -> None:
        """Parse bond block in the sdf file.
        Save undirected edges(`self.bond_index`) and their types (`self.bond_type`) as an instance variable.
        Edges are sorted in ascending order.

        Args:
            bond_block (list[str]): bond information block, parsed with `splitlines()`. Each string represents a bond.
        """
        # BondType -> str : BondType.AROMATIC.name = AROMATIC
        # BondType -> int : int(BondType.AROMATIC) = 12
        # int -> BondType : BondType.values[12] = BondType.AROMATIC
        self.bond_index = torch.empty((2, 2 * self.num_bonds), dtype=torch.int)
        self.bond_type = torch.empty(2 * self.num_bonds, dtype=torch.int)
        for idx, bond_line in enumerate(bond_block):
            start, end = int(bond_line[0:3]) - 1, int(bond_line[3:6]) - 1
            bond_type_sdf = int(bond_line[6:9])
            bond_type_rdkit = BOND_SDF_TO_RDKIT[bond_type_sdf]
            # insert two edges to implement an undirected edge.
            self.bond_index[:, 2 * idx : 2 * (idx + 1)] = torch.tensor([[start, end], [end, start]])
            self.bond_type[2 * idx : 2 * (idx + 1)] = torch.tensor([bond_type_rdkit, bond_type_rdkit])

        # group edges with same source node and sort them in ascending order.
        perm = (self.bond_index[0] * self.num_atoms + self.bond_index[1]).argsort()
        self.bond_index = self.bond_index[:, perm]
        self.bond_type = self.bond_type[perm]

    def parse_sdf(self, sdf_path: Union[str, Path]) -> None:
        """parse ligand file in sdf format.

        Args:
            sdf_path (Union[str, Path]): path to the sdf file.
        """
        with open(sdf_path, "r") as file:  # pylint: disable=unspecified-encoding
            sdf = file.read().splitlines()

        self.num_atoms, self.num_bonds = int(sdf[3][0:3]), int(sdf[3][3:6])

        self._get_SMARTS_feature_matrix(sdf_path)
        self._parse_atoms(sdf[4 : 4 + self.num_atoms])
        self._parse_bonds(sdf[4 + self.num_atoms : 4 + self.num_atoms + self.num_bonds])
