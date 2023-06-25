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
    # print(smiles)
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

def get_BRICS_bonds(molecule):
    brics= Chem.BRICS.FindBRICSBonds(molecule, randomizeOrder=False, silent=True)

    return [i for i in brics]

def break_bond(molecule,bonds:list):
    emol =  Chem.EditableMol(molecule)

    for bond in bonds:
        emol.RemoveBond(bond[0],bond[1])
    back = emol.GetMol()
    Chem.SanitizeMol(back)

    frags = Chem.GetMolFrags(back,asMols=True)
    mols = [i for i in frags]

    return mols

    

def get_all_bonds_idxs(molecule):
    bonds = molecule.GetBonds()
    cliques = []
    for bond in bonds:
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1,a2])

    return cliques


def find_overlap_with_BRICS(molecule,bonds):
    BRICSbondsWithDummy = get_BRICS_bonds(molecule)
    BRICSbonds = [i[0] for i in BRICSbondsWithDummy]
    bonds_map = [0]*len(bonds)

    filteredBonds= [ ]

    for i in range(len(bonds)):
        if tuple(bonds[i]) in BRICSbonds or tuple([bonds[i][1],bonds[i][0]]) in BRICSbonds:
            bonds_map[i]=1

            filteredBonds.append(bonds[i])

    return bonds_map,filteredBonds