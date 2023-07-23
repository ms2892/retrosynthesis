from rdkit import Chem
from rdkit.Chem import BRICS
from paroutes import PaRoutesInventory, PaRoutesModel, get_target_smiles
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs
from rdkit.Chem import RDConfig
import os
import sys
import numpy as np
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer

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
    try:
        emol =  Chem.EditableMol(molecule)

        for bond in bonds:
            emol.RemoveBond(bond[0],bond[1])
        back = emol.GetMol()
        Chem.SanitizeMol(back)

        frags = Chem.GetMolFrags(back,asMols=True)
        mols = [i for i in frags]

        return mols
    except:
        return [molecule]

    

def get_all_bonds_idxs(molecule):
    bonds = molecule.GetBonds()
    cliques = []
    for bond in bonds:
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1,a2])
    cliques.sort()
    return cliques

def get_fingerprint( mol: AllChem.Mol):
    return AllChem.GetMorganFingerprint(mol, radius=3)


def calculate_estimate(mols,inventory):
    nn_dist = []
    smiles_query = [Chem.MolToSmiles(i) for i in mols]
    fp_mols =list(map(AllChem.MolFromSmiles,inventory))
    fps_inv = list(map(get_fingerprint,fp_mols))
    for i in smiles_query:
        fp_query = get_fingerprint(AllChem.MolFromSmiles(i))
        tanimoto_sims = DataStructs.BulkTanimotoSimilarity(fp_query,fps_inv)
        nn_dist.append(max(tanimoto_sims))
    return min(nn_dist)

def reward_func(molecules,mode,inventory=None):
    if mode=='sas':
        sum_sa = []
        for i in molecules:
            sum_sa.append((10 - sascorer.calculateScore(i))/10)
        return min(sum_sa)
    
    elif mode=='nn' and inventory is not None:
        nn_dist = []
        smiles_query = [Chem.MolToSmiles(i) for i in molecules]
        fps_mols = list(map(AllChem.MolFromSmiles, inventory))
        fps_inv = list(map(get_fingerprint, fps_mols))
        for i in smiles_query:
            fp_query = get_fingerprint(AllChem.MolFromSmiles(i))
            tanimoto_sims = DataStructs.BulkTanimotoSimilarity(fp_query, fps_inv)
            nn_dist.append(max(tanimoto_sims))
        return min(nn_dist)


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




class SASEnvironmentRAND:
    def __init__(self,inventory:list):
        self.smile_db = inventory
        self.action_space = self._get_max_bonds
        self.inventory = get_target_smiles(5)
        # self.observation_space = len(self.smile_db)

    def _get_max_bonds(self):
        mxm_bonds=-1
        for smile in self.smile_db:
            molecule = Chem.MolFromSmiles(smile)
            bonds = get_all_bonds_idxs(molecule)
            mxm_bonds = max(mxm_bonds,len(bonds))

        return mxm_bonds
    
    def get_inventory(self):
        return self.new_inventory
    
    def step(self,bonds,smile,rand=True):
        BRICS = get_BRICS_bonds(Chem.MolFromSmiles(smile))
        mol = Chem.MolFromSmiles(smile)
        if len(BRICS)==1:
            return [mol], reward_func([mol],mode='nn',inventory=self.new_inventory),True
        mols = break_bond(mol,bonds)
        reward = reward_func(mols,mode='nn',inventory=self.new_inventory)

        done_flag=False
        # print(reward)
        self.new_inventory = np.random.choice(self.inventory,int(len(self.inventory)*0.9),replace=False)
        return mols,reward,done_flag

class SASEnvironment:
    def __init__(self,inventory:list):
        self.smile_db = inventory
        self.action_space = self._get_max_bonds
        self.inventory = get_target_smiles(5)
        # self.observation_space = len(self.smile_db)

    def _get_max_bonds(self):
        mxm_bonds=-1
        for smile in self.smile_db:
            molecule = Chem.MolFromSmiles(smile)
            bonds = get_all_bonds_idxs(molecule)
            mxm_bonds = max(mxm_bonds,len(bonds))

        return mxm_bonds
    
    def get_inventory(self):
        return self.inventory
    
    def step(self,bonds,smile,rand=False):
        BRICS = get_BRICS_bonds(Chem.MolFromSmiles(smile))
        mol = Chem.MolFromSmiles(smile)
        if not rand:
            new_inventory = np.random.choice(self.inventory,int(len(self.inventory)*0.9),replace=False)
        else:
            new_inventory = self.inventory
        if len(BRICS)==1:
            return [mol], reward_func([mol],mode='nn',inventory=new_inventory),True
        mols = break_bond(mol,bonds)
        reward = reward_func(mols,mode='nn',inventory=new_inventory)

        done_flag=False
        # print(reward)
        return mols,reward,done_flag