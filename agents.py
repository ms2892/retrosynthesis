import rdkit
from rdkit import Chem
from split_environment import *
import random
from GNN_utils import *
from GNN_agent import *

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

class GNNAgent:
    def __init__(self):
        temp_smiles = 'CC(=O)OC1CCCC1(N)CNc1ccc(C#N)c(Cl)c1C'
        temp_molecule = Chem.MolFromSmiles(temp_smiles)
        temp_graph = graph_from_molecule(temp_molecule)
        atom_dim = temp_graph[0].shape[-1]
        bond_dim = temp_graph[1].shape[-1]
        print(bond_dim,atom_dim)
        self.GNNModel = MPNNModel(atom_dim=atom_dim,bond_dim=bond_dim,message_units=64,message_steps=4)

    
    def action(self,smiles:str):
        molecule = Chem.MolFromSmiles(smiles)
        graph = graphs_from_smiles([smiles])
        

        y = np.zeros((1))
        data = MPNNDataset(graph,y)
        bonds = get_all_bonds_idxs(molecule)

        atom_feats = self.GNNModel.predict(data)

        # atom_feats = atom_feats.numpy()

        edge_embeddings = []
        
        for bond in bonds:
            atm1 = bond[0]
            atm2 = bond[1]

            bond_embeddings = atom_feats[atm1,:]+atom_feats[atm2,:]

            edge_embeddings.append(bond_embeddings)

        return edge_embeddings,bonds
            
        

    def update():
        pass