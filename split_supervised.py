from rdkit import *
from split_environment import *
from glob import glob
from multiprocessing import Pool

from neighbour_value_functions import *
from tqdm import tqdm
import itertools
import json

all_files = glob('splits/*.json')

all_files = [i.split('/')[1] for i in all_files]
all_files = [i[:-5] for i in all_files]
all_files.sort()

def get_output(smile):
    if smile in all_files:
        print(f'Found smile {smile} exists. Not Calculating this shit')
        return 
    else:
        print("Didn't find the smile")
    # t=input()
    bonds = get_all_bonds_idxs(Chem.MolFromSmiles(smile))
    with open(f'purchasable/{smile}.json','r') as f:
        configDict= json.load(f)

    cutBonds = []
    if not configDict:
        print("Skipping. No data present")
        return 
    print("Total Purchasable Mols: ",len(configDict[smile]))
    for i in configDict[smile][:20]:
        num_cuts = len(i)-1
        combinations = itertools.combinations(bonds,num_cuts)
        mnm_sims = float('inf')
        mnm_bonds = []
        for cuts in combinations:
            mols = break_bond(Chem.MolFromSmiles(smile),cuts)
            if len(mols)==len(i):
                smile_frags = [Chem.MolToSmiles(j) for j in mols]

                similarityClass = TanimotoCalculator(i)

                sims = [similarityClass._get_nearest_neighbour_dist(j) for j in smile_frags]
                if np.mean(sims)<mnm_sims:
                    mnm_sims = np.mean(sims)
                    mnm_bonds = cuts
        print(f'{smile} appended +1')
        cutBonds.append(mnm_bonds)

    from collections import defaultdict
    bondDict=defaultdict(set)

    for cuts in cutBonds:
        bondMap = [0]*len(bonds)
        for cut in cuts:
            index = bonds.index(cut)
            bondMap[index]=1
        bondDict[smile].add(tuple(bondMap))

    bondDict[smile] = [list(i) for i in bondDict[smile]]

    with open(f'splits/{smile}.json','w') as f:
        json.dump(bondDict,f)
    print(f'{smile} SAVED')


def save_outputs(smiles):
    with Pool(8) as p:
        p.map(get_output,smiles)
    # for i in smiles:
        # get_output(i,all_files)





if __name__=='__main__':
    smile_files = glob('purchasable/*.json')
    smiles=[]
    for smile_file in smile_files:
        smile = smile_file.split('/')[1]
        smiles.append(smile[:-5])
    # print(all_files)

    smiles.sort()
    # print(all_files)
    save_outputs(smiles)
