# pylint: disable=redefined-outer-name,missing-module-docstirng,pointless-statement
from pathlib import Path

import pytest

from molecule_parser.protein_parser import ProteinParser


@pytest.fixture
def protein_parser() -> ProteinParser:
    """Fixture for ProteinParser."""
    return ProteinParser()


def test_before_parsing(protein_parser: ProteinParser):
    """Accessing parsed data before parsing should raise an error."""
    with pytest.raises(ValueError):
        protein_parser.header
    with pytest.raises(ValueError):
        protein_parser.atom_data
    with pytest.raises(ValueError):
        protein_parser.residue_data


def test_protein_parser(protein_parser: ProteinParser):
    """Test parsing a pdb file.

    pdb REMARK data

    line 75-76
    REMARK   3  NUMBER OF NON-HYDROGEN ATOMS USED IN REFINEMENT.
    REMARK   3   PROTEIN ATOMS            : 8045

    line 261-443
    missing residues chain A : residue_id 1-24, 526-584
                     chain B : residue_id 1-24, 476-584
    chain A : 25-525 (501 residues)
    chain B : 25-475, 489-527 (490 residues)
    """
    pdb_path = Path(__file__).parent / "sample.pdb"
    protein_parser.parse_pdb(pdb_path)
    assert protein_parser.header == "TRANSFERASE                             27-FEB-15   4YHJ"

    assert protein_parser.atom_data.atom_name.shape == (8045,)
    assert protein_parser.atom_data.aa_type.shape == (8045,)
    assert protein_parser.atom_data.atomic_numbers.shape == (8045,)
    assert protein_parser.atom_data.is_backbone.shape == (8045,)
    assert protein_parser.atom_data.pos.shape == (8045, 3)

    assert protein_parser.residue_data.aa_type.shape == (991,)
    assert protein_parser.residue_data.center_of_mass.shape == (991, 3)
    assert protein_parser.residue_data.pos_CA.shape == (991, 3)
    assert protein_parser.residue_data.pos_C.shape == (991, 3)
    assert protein_parser.residue_data.pos_N.shape == (991, 3)
    assert protein_parser.residue_data.pos_O.shape == (991, 3)
