"""Module for parsing ligand data with '.sdf' format."""
# %%
import typing
from pathlib import Path
from typing import Union

import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

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


class LigandParser:
    """Parser for ligand data with '.sdf' format."""

    def __init__(self) -> None:
        fdef_name = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
        self.factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))

    def parse_sdf(self, sdf_path: Union[str, Path]) -> None:
        """TODO: Add docstrint"""
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        # assume a sdf file includes a single ligand
        mol = next(supplier)
        # construct feature matrix
        rd_num_atoms = mol.GetNumAtoms()
        feat_mat = torch.zeros((rd_num_atoms, len(ATOM_FAMILIES)), dtype=torch.long)
        # Extracting SMARTS-based features using RDKit
        # Corresponding document: https://www.rdkit.org/docs/GettingStartedInPython.html#chemical-features
        for feature in self.factory.GetFeaturesForMol(mol):
            feat_mat[feature.GetAtomIds(), ATOM_FAMILIES_ID[feature.GetFamily()]] = 1

        with open(sdf_path, "r") as f:
            sdf = f.read().splitlines()

        num_atoms, num_bonds = int(sdf[3][0:3]), int(sdf[3][3:6])
        assert num_atoms == rd_num_atoms, "Number of atoms from RDkit and SDF mismatch!"
