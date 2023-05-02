# pylint: disable=redefined-outer-name,missing-module-docstirng,pointless-statement
from pathlib import Path

import pytest

from molecule_parser.ligand_parser import LigandParser


@pytest.fixture
def ligand_parser() -> LigandParser:
    """Fixture for LigandParser."""
    return LigandParser()


def test_ligand_parser(ligand_parser: LigandParser):
    sdf_path = Path(__file__).parent / "sample.sdf"
    ligand_parser.parse_sdf(sdf_path)

    assert ligand_parser.num_atoms == 27
    assert ligand_parser.num_bonds == 29
    assert ligand_parser.feat_mat.shape == (ligand_parser.num_atoms, 8)
    assert ligand_parser.positions.shape == (ligand_parser.num_atoms, 3)
    assert ligand_parser.elements.shape == (ligand_parser.num_atoms,)
    assert ligand_parser.center_of_mass.shape == (3,)
    assert ligand_parser.bond_index.shape == (2, 2 * ligand_parser.num_bonds)
    assert ligand_parser.bond_type.shape == (2 * ligand_parser.num_bonds,)
