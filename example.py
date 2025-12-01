import csv
from pathlib import Path

from rdkit import RDLogger
from canonical_featurizers import featurize_molecule

# Silence noisy RDKit deprecation warnings like GetValence(getExplicit=False)
RDLogger.DisableLog("rdApp.warning")

THIS_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = THIS_DIR / "example_smiles.csv"

# Optional human-readable names for a few example SMILES
MOLECULE_NAMES = {
    "CC": "ethane",
    "CCC": "propane",
    "CCCC": "butane",
    "CCO": "ethanol",
    "c1ccccc1": "benzene",
    "CC(=O)O": "acetic acid",
}


def load_smiles(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Expected CSV file not found: {path}\n"
            "Put a small 100-SMILES file named 'example_smiles.csv' next to example.py.\n"
            "CSV header must be: id,smiles"
        )

    smiles_list = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append((row["id"], row["smiles"]))
    return smiles_list


def main():
    # 1) Load SMILES from existing CSV
    smiles_data = load_smiles(SAMPLE_CSV)
    print(f"Loaded {len(smiles_data)} SMILES from {SAMPLE_CSV.name}")

    # 2) Featurize each molecule
    all_atom_feat_shapes = []
    all_bond_feat_shapes = []
    example_details = []  # store info for up to 5 molecules

    for idx, smi in smiles_data:
        try:
            atom_feats, bond_feats = featurize_molecule(smi)
        except ValueError as e:
            print(f"[WARN] Skipping row id={idx} due to parsing error: {e}")
            continue

        atom_shape = atom_feats["h"].shape
        bond_shape = bond_feats["e"].shape

        all_atom_feat_shapes.append(atom_shape)
        all_bond_feat_shapes.append(bond_shape)

        if len(example_details) < 5:
            name = MOLECULE_NAMES.get(smi, "(name not set)")
            example_details.append(
                {
                    "id": idx,
                    "smiles": smi,
                    "name": name,
                    "atom_shape": atom_shape,
                    "bond_shape": bond_shape,
                }
            )

    # 3) Print a tiny summary
    if not all_atom_feat_shapes:
        print("No valid SMILES were featurized.")
        return

    print("\nExample featurization summary")
    print("-----------------------------------")
    print(f"Number of molecules featurized: {len(all_atom_feat_shapes)}")
    first_atom_shape = all_atom_feat_shapes[0]
    first_bond_shape = all_bond_feat_shapes[0]
    print(f"Atom feature shape of first molecule: {first_atom_shape}  (N_atoms, 74)")
    print(f"Bond feature shape of first molecule: {first_bond_shape}  (N_bonds*2, 12)")

    # 4) Show details for up to 5 example molecules
    print("\nExample molecules (up to 5):")
    for ex in example_details:
        print(
            f"  id={ex['id']}, name={ex['name']}, "
            f"smiles={ex['smiles']}, "
            f"atoms_shape={ex['atom_shape']}, bonds_shape={ex['bond_shape']}"
        )


if __name__ == "__main__":
    main()


