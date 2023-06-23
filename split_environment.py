from rdkit import Chem
from rdkit.Chem import BRICS

def split_molecule(smile:str):
    return set(Chem.BRICS.BRICSDecompose(Chem.MolFromSmiles(smile)))

def split_molecule_batch(smiles:list):
    result = []
    for smile in smiles:
        result.append(split_molecule(smile))
    return result

def convert_to_mol_batch(smiles:list):
    result = []
    print(smiles)
    for smile in smiles:
        result.append(Chem.MolFromSmiles(smile))
    return result

def convert_splits_to_mol_batch(split_smiles:list):
    result = []

    for split_mols in split_smiles:
        splits_mols = []
        for split_smiles in split_mols:
            splits_mols.append(Chem.MolFromSmiles(split_smiles))

        result.append(splits_mols)

    return result