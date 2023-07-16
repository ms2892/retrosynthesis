import os

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import GATConv    # official GAT implementation in PyG
from torch_geometric.datasets import Planetoid 
import torch_geometric.transforms as T
from rdkit import RDLogger
import torch_geometric
from split_environment import *
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage

# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

np.random.seed(42)
tf.random.set_seed(42)


class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()

def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule




atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)


def graph_from_molecule(molecule):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        # Add self-loops
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)

def prepare_batch(x_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph
    """

    atom_features, bond_features, pair_indices = x_batch

    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    # Obtain partition indices (molecule_indicator), which will be used to
    # gather (sub)graphs from global graph in model later on
    molecule_indices = tf.range(len(num_atoms))
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)

    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch


def MPNNDataset(X, y, batch_size=32, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)


def graphs_from_smiles(smiles_list):
    # Initialize graphs
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for smiles in smiles_list:
        molecule = molecule_from_smiles(smiles)
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)

    # Convert lists to ragged tensors for tf.data.Dataset later on
    return (
        tf.ragged.constant(atom_features_list, dtype=tf.float32),
        tf.ragged.constant(bond_features_list, dtype=tf.float32),
        tf.ragged.constant(pair_indices_list, dtype=tf.int64),
    )



def convert_smiles_to_pyg_graph(smiles):
    return torch_geometric.utils.from_smiles(smiles)

class GAT(nn.Module):
    def __init__(self,in_features,out_features):
        super(GAT,self).__init__()
        self.hid = 64
        self.in_head = 64
        self.out_head = 5
        
        self.conv1 = GATConv(in_features,self.hid,heads=self.in_head,dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head,out_features,concat=False,heads=self.out_head,dropout=0.6)
        
    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        x =x.float()
        x = F.dropout(x,p=0.6,training=self.training)
        x = self.conv1(x,edge_index)
        x = F.elu(x)
        x = F.dropout(x,p=0.6,training=self.training)
        x = self.conv2(x,edge_index)
        
        bonds = get_BRICS_bonds(Chem.MolFromSmiles(data['smiles']))

        bonds = [list(i[0]) for i in bonds] 
        bonds.sort()
        # t=input()   
        lefts=[]
        rights=[]

        for i in bonds:
            lefts.append(i[0])
            rights.append(i[1])
        lefts = np.array(lefts)
        rights = np.array(rights)
        lefts = torch.from_numpy(lefts)
        rights = torch.from_numpy(rights)

        # src, dst = edge_index
        # print(src,dst)
        score = torch.sum(x[lefts] + x[rights],dim=-1)
        # print(score.shape)  
        return F.softmax(score,dim=0)

class GATCritic(nn.Module):
    def __init__(self,in_features,out_features):
        super(GATCritic,self).__init__()
        self.hid = 64
        self.in_head = 64
        self.out_head = 5
        
        self.conv1 = GATConv(in_features,self.hid,heads=self.in_head,dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head,out_features,concat=False,heads=self.out_head,dropout=0.6)
        
    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        x =x.float()
        x = F.dropout(x,p=0.6,training=self.training)
        x = self.conv1(x,edge_index)
        x = F.elu(x)
        x = F.dropout(x,p=0.6,training=self.training)
        x = self.conv2(x,edge_index)  
        s = torch.mean(x,dim=0)
        return F.tanh(s)

if __name__=='__main__':
    model = GAT(9,32)
    mol_graph = convert_smiles_to_pyg_graph('CS(=O)(=O)NC(=O)c1cc(C2CC2)c(OCC2(C#N)C3CC4CC(C3)CC2C4)cc1F')
    print(model(mol_graph))
    print(convert_smiles_to_pyg_graph('CS(=O)(=O)NC(=O)c1cc(C2CC2)c(OCC2(C#N)C3CC4CC(C3)CC2C4)cc1F'))