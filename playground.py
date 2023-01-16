# %%
from pathlib import Path

import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

# %%
fdef_name = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
sdf_path = Path.cwd() / "molecule_parser" / "sample.sdf"
factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))
supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
# %%
mol = next(supplier)
# %%
acc = 0
for i, feature in enumerate(factory.GetFeaturesForMol(mol)):
    print("========")
    print(feature.GetFamily())
    print(feature.GetAtomIds())
    print(list(feature.GetPos()))
# %%
from rdkit.Chem import AllChem

# %%
