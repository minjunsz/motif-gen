"""Module for parsing protein data with '.pdb' format."""
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np

from .pdb_dtypes import (
    PDBAtomData,
    PDBAtomLine,
    PDBAtomNumpy,
    PDBHeaderLine,
    PDBResidueNumpy,
    PDBUnhandledLine,
    get_residue_signature,
)


class ProteinParser:
    """Parser for protein data with '.pdb' format."""

    _header: str
    _atom_data_numpy: PDBAtomNumpy
    _residue_data_numpy: PDBResidueNumpy

    @property
    def header(self) -> str:
        """get parsed header."""
        if not hasattr(self, "_header"):
            raise ValueError("Must parse a pdb file first.")
        return self._header

    @property
    def atom_data(self) -> PDBAtomNumpy:
        """get parsed numpy atom data."""
        if not hasattr(self, "_atom_data_numpy"):
            raise ValueError("Must parse a pdb file first.")
        return self._atom_data_numpy

    @property
    def residue_data(self) -> PDBResidueNumpy:
        """get parsed numpy residue data."""
        if not hasattr(self, "_residue_data_numpy"):
            raise ValueError("Must parse a pdb file first.")
        return self._residue_data_numpy

    def parse_pdb(self, pdb_path: Union[str, Path]) -> None:
        """parse a pdb file. Parsed data is stored as `self.header` and `self.atom_data_numpy`."""
        # load a pdb file
        path = Path(pdb_path)
        if path.suffix.lower() != ".pdb":
            raise ValueError("Must provide a path to '.pdb' file")
        if not path.exists():
            raise ValueError(f"File {path.name} does not exist.")

        atom_data = PDBAtomData()
        atom_index: int = 0

        # parse information line by line.
        signature_to_residue: dict[str, list[int]] = defaultdict(list)
        for line in path.open("r"):
            parsed_line = self._parse_pdb_line(line)
            if parsed_line["type"] == "HEADER":
                self._header = parsed_line["value"]
                continue
            if parsed_line["type"] == "UNHANDLED":
                continue

            # parsed_line is PDBAtomLine
            atom_data.add_atom(parsed_line)
            residue_signature = get_residue_signature(parsed_line)
            signature_to_residue[residue_signature].append(atom_index)
            atom_index += 1

        self._residue_data_numpy = PDBResidueNumpy(len(signature_to_residue))
        # save information of residues
        for res_idx, residue in enumerate(signature_to_residue.values()):
            residue_atoms_idx = np.array(residue)

            # save residue type
            amino_acid_type = np.array(atom_data.atom_aa_index)[residue_atoms_idx][0]
            self._residue_data_numpy.aa_type[res_idx] = amino_acid_type

            # calculate center of mass
            weights = np.array(atom_data.atomic_weight)[residue_atoms_idx]
            positions = np.array(atom_data.pos)[residue_atoms_idx]
            center_of_mass = (positions * weights[:, None]).sum(axis=0) / weights.sum(axis=0)
            self._residue_data_numpy.center_of_mass[res_idx] = center_of_mass

            # save backbone positions (N, CA, C, O)
            atom_name = np.array(atom_data.atom_name)[residue_atoms_idx]
            is_backbone = np.array(atom_data.is_backbone)[residue_atoms_idx]
            backbone_pos = positions[is_backbone]
            backbone_name = atom_name[is_backbone]
            assert len(backbone_pos) == 4
            for i in range(4):
                if backbone_name[i] == "N":
                    self._residue_data_numpy.pos_N[res_idx] = backbone_pos[i]
                elif backbone_name[i] == "CA":
                    self._residue_data_numpy.pos_CA[res_idx] = backbone_pos[i]
                elif backbone_name[i] == "C":
                    self._residue_data_numpy.pos_C[res_idx] = backbone_pos[i]
                elif backbone_name[i] == "O":
                    self._residue_data_numpy.pos_O[res_idx] = backbone_pos[i]
                else:
                    raise NotImplementedError(f"backbone atom {backbone_name[i]} is not implemented.")

        self._atom_data_numpy = PDBAtomNumpy(atom_data)

    def _parse_pdb_line(self, line: str) -> Union[PDBAtomLine, PDBHeaderLine, PDBUnhandledLine]:
        line_type = line[0:6].strip()
        match line_type:
            case "ATOM":
                element_symb = line[76:78].strip().capitalize()
                element_symb = line[13:14] if element_symb == "" else element_symb
                atom_info: PDBAtomLine = {
                    "line": line,
                    "type": "ATOM",
                    "atom_id": int(line[6:11]),
                    "atom_name": line[12:16].strip(),
                    "residue_name": line[17:20].strip(),
                    "chain_id": line[21:22].strip(),
                    "residue_id": int(line[22:26]),
                    "residue_insert_id": line[26:27].strip(),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "occupancy": float(line[54:60]),
                    "segment_id": line[72:76].strip(),
                    "element_symb": element_symb,
                    "charge": line[78:80].strip(),
                }
                return atom_info
            case "HEADER":
                return {"type": "HEADER", "value": line[10:].strip()}
            case "ENDMDL":
                raise StopIteration
            case _:
                return {"type": "UNHANDLED"}
