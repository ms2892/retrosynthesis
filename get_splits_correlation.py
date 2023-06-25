import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from functools import partial
from collections import defaultdict
import pandas as pd
from rdkit.Chem import BRICS
from rdkit.Chem import AllChem
from rdkit import DataStructs


from paroutes import PaRoutesInventory, PaRoutesModel, get_target_smiles

def calc_tanimoto(smile1,smile2):
    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1,2,nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2,2,nBits=2048)

    s = round(DataStructs.TanimotoSimilarity(fp1,fp2),3)
    return s

def get_correlation(target_molecules:list,purchasable_mols:list):
    for smile in target_molecules:
        molecules = set(Chem.BRICS.BRICSDecompose(Chem.MolFromSmiles(smile)))
        sims = []
        for mol in molecules:
            mxm_sim= 0
            for pur_mol in purchasable_mols:
                try:
                    mxm_sim = max(mxm_sim,calc_tanimoto(mol,pur_mol.smiles))
                    if mxm_sim==1:
                        break
                except:
                    print(mol)
                    break
            sims.append((mol,mxm_sim))
        mxm_index=-1
        mxm_val=-1
        for i in range(len(sims)):
            if sims[i][1]>mxm_val:
                mxm_index = i
                mxm_val = sims[i][1]

        if len(sims)!=0:
            avg_res = sum([i[1] for i in sims])/len(sims)

            temp = [smile,str(avg_res),sims[mxm_index][0],str(sims[mxm_index][1]),'\n']
            with open('para_sims.csv','a') as f:
                f.write(', '.join(temp))