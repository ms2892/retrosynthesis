import rdkit
from rdkit import Chem
from split_environment import *
import random

class RandomAgent:

    def __init__(self):
        pass

    def action(self,smiles:str,cuts=1):
        molecule = Chem.MolFromSmiles(smiles)
        bonds = get_all_bonds_idxs(molecule)
        random_bond_idx = random.randint(0,len(bonds)-1)
        return [bonds[random_bond_idx]]

    def update(self):
        pass

class EqualSplitAgent:
    def __init__(self):
        pass

    def get_count_atms(self,molecule):
        cnt=0
        for a in molecule.GetAtoms():
            cnt+=1
        return cnt

    def get_difference(self,molecule,bond):
        try:
            mols = break_bond(molecule,[bond])
            mol1 = mols[0]
            mol2 = mols[1]
            return abs(self.get_count_atms(mol1) - self.get_count_atms(mol2))
        except:
            return abs(self.get_count_atms(molecule))

    def action(self,smiles:str):
        molecule = Chem.MolFromSmiles(smiles)
        bonds = get_all_bonds_idxs(molecule)
        mnm_diff=float('inf')
        mnm_index = -1
        for i in range(len(bonds)):
            curr_diff = self.get_difference(molecule,bonds[i])
            if mnm_diff>curr_diff:
                mnm_diff = curr_diff
                mnm_index = i

        return [bonds[mnm_index]]

    def update():
        pass