# SMILES to Graphs

This mini-project exists because DGL-LifeSci and similar graph stacks were not
being handled reliably in my environments. I wanted a simple, self-contained
way to featurize SMILES strings without depending on those heavier libraries.

A ready-to-use `canonical_featurizers.py` that gives you:

- `CanonicalAtomFeaturizer` → 74-dim atom vectors (type, degree, aromaticity …)
- `CanonicalBondFeaturizer` → 12-dim bond vectors + edge indices
- `featurize_molecule(smiles)` helper → ready-to-consume arrays that you can
  pass into whatever graph library you prefer (or your own code).

## Clone-and-run (as its own GitHub project)

If you publish this as a standalone repo (for example `SMILES-to-graphs`), a
typical workflow for new users would be:

```bash
git clone https://github.com/<your-username>/SMILES-to-graphs.git
cd SMILES-to-graphs

# install minimal deps (ideally inside a virtualenv)
pip install rdkit numpy

# run the demo script
python example.py

# or open the notebook
jupyter notebook example.ipynb
```

## How to use it inside another repo

1. **Copy the folder** (or just the `.py` file) into your project.
2. **Install prerequisites**
   ```bash
   pip install rdkit numpy
   ```
3. **Import and featurize**
   ```python
   from canonical_featurizers import featurize_molecule

   atom_feats, bond_feats = featurize_molecule("CCO")
   print(atom_feats["h"].shape)        # (num_atoms, 74)
   print(bond_feats["edge_indices"])   # (2, num_edges)
   ```
4. Wire the outputs into your favourite graph neural network library.

## Troubleshooting

- **RDKit missing?** Use `pip install rdkit` or `conda install -c conda-forge rdkit`.
- **SMILES parsing fails?** RDKit returns `None`; wrap `Chem.MolFromSmiles`
  with a check and raise a helpful error (already done in the script).
- **Need self-loops or padding?** Control it through the `self_loop` argument
  when constructing `CanonicalBondFeaturizer`.





