"""Hold code for heuristic value functions."""

from collections.abc import Sequence
from typing import Optional
from rdkit import Chem
from rdkit.Chem import RDConfig
import os
from rdkit import *
import torch
import sys
from GNN_utils import *
from rdkit.Chem import DataStructs, AllChem

from split_environment import *
from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from syntheseus.search.graph.and_or import OrNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator


class ScaledSAScoreCostFunction(NoCacheNodeEvaluator[OrNode]):
    """
    Estimates the cost to make a molecule as alpha * synthetic accessability score,
    noting that higher SA score generally means harder to make.
    """

    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
        sa_scores = [
            sascorer.calculateScore(Chem.MolFromSmiles(node.mol.smiles))
            for node in nodes
        ]
        return [self.scale * v for v in sa_scores]


class GNNEstimatorFunction(NoCacheNodeEvaluator):

    def __init__(self,scale,inventory,mode='max',**kwargs):
        super().__init__(**kwargs)
        self.scale=scale
        self.model = torch.load('GNN_split_Model.pth')
        self._set_fingerprints([mol.smiles for mol in inventory.purchasable_mols()])
        self.mode = mode
    def get_fingerprint(self, mol: AllChem.Mol):
        return AllChem.GetMorganFingerprint(mol, radius=3)

    def _set_fingerprints(self, smiles_list: list[str]) -> None:
        """Initialize fingerprint cache."""
        mols = list(map(AllChem.MolFromSmiles, smiles_list))
        assert None not in mols, "Invalid SMILES encountered."
        self._fps = list(map(self.get_fingerprint, mols))

    def _get_nearest_neighbour_dist(self, smiles: str) -> float:
        fp_query = self.get_fingerprint(AllChem.MolFromSmiles(smiles))
        tanimoto_sims = DataStructs.BulkTanimotoSimilarity(fp_query, self._fps)
        return 1 - max(tanimoto_sims)

    def get_nn_dist(self,smiles):
        pyg_graph = convert_smiles_to_pyg_graph(smiles)
        brics = get_BRICS_bonds(Chem.MolFromSmiles(smiles))
        if brics:
            edges = self.model(pyg_graph)
            action = np.argmax(edges.detach().numpy())
            bond = brics[action]

            mols = break_bond(Chem.MolFromSmiles(smiles),[bond])
            smiles_mols = [Chem.MolToSmiles(i) for i in mols]
            nn_dist = [self._get_nearest_neighbour_dist(smile) for smile in smiles_mols]
            
            if self.mode=='max':
                return np.max(nn_dist)
            elif self.mode=='avg':
                return np.mean(nn_dist)
        else:
            return self._get_nearest_neighbour_dist(smiles)

    def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
        nn_dist = [
            self.get_nn_dist(node.mol.smiles)
            for node in nodes
        ]
        return [self.scale*v for v in nn_dist]
