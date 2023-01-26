"""Module for parsing protein data with '.pdb' format."""
from pathlib import Path
from typing import Union

from rdkit import Chem


class ProteinParser:
    """Parser for protein data with '.pdb' format."""

    def __init__(self):
        self.ptable = Chem.GetPeriodicTable()

    def parse_pdb(self, pdb_path: Union[str, Path]) -> None:
        pass
