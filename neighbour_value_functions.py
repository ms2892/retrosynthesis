"""Basic code for nearest-neighbour value functions."""
from __future__ import annotations

from enum import Enum

import numpy as np
from rdkit.Chem import DataStructs, AllChem
from GNN_utils import *
from syntheseus.search.graph.and_or import OrNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator
from syntheseus.search.mol_inventory import ExplicitMolInventory


class DistanceToCost(Enum):
    NOTHING = 0
    EXP = 1
    QUAD = 2
    CUB = 3
    SIN = 4
    LOG = 5


class TanimotoCalculator:
    """Estimates cost of a node using Tanimoto distance to purchasable molecules."""

    def __init__(
        self,
        inventory: ExplicitMolInventory,
        # distance_to_cost: DistanceToCost,
        # **kwargs,
    ):
        # super().__init__(**kwargs)
        # self.distance_to_cost = distance_to_cost
        self._set_fingerprints([mol for mol in inventory])

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

class TanimotoNNCostEstimator(NoCacheNodeEvaluator):
    """Estimates cost of a node using Tanimoto distance to purchasable molecules."""

    def __init__(
        self,
        inventory: ExplicitMolInventory,
        distance_to_cost: DistanceToCost,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.distance_to_cost = distance_to_cost
        self._set_fingerprints([mol.smiles for mol in inventory.purchasable_mols()])

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

    def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
        if len(nodes) == 0:
            return []

        # Get distances to nearest neighbours
        nn_dists = np.asarray(
            [self._get_nearest_neighbour_dist(node.mol.smiles) for node in nodes]
        )
        assert np.min(nn_dists) >= 0

        # Turn into costs
        if self.distance_to_cost == DistanceToCost.NOTHING:
            values = nn_dists
        elif self.distance_to_cost == DistanceToCost.EXP:
            values = np.exp(nn_dists) - 1
        elif self.distance_to_cost == DistanceToCost.QUAD:
            values = np.power(nn_dists,2)
        elif self.distance_to_cost == DistanceToCost.CUB:
            values = np.power(nn_dists,3)
        elif self.distance_to_cost == DistanceToCost.SIN:
            values = np.sin(nn_dists)
        elif self.distance_to_cost == DistanceToCost.LOG:
            values = np.log(1 + nn_dists)
        else:
            raise NotImplementedError(self.distance_to_cost)

        return list(values)


class DiceNNCostEstimator(NoCacheNodeEvaluator):
    '''Estimate the cost of Nearest neighbor using Dice distance function'''

    def __init__(
        self,
        inventory: ExplicitMolInventory,
        distance_to_cost: DistanceToCost,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.distance_to_cost = distance_to_cost
        self._set_fingerprints([mol.smiles for mol in inventory.purchasable_mols()])

    def get_fingerprint(self, mol: AllChem.Mol):
        return AllChem.GetMorganFingerprint(mol, radius=3)

    def _set_fingerprints(self, smiles_list: list[str]) -> None:
        """Initialize fingerprint cache."""
        mols = list(map(AllChem.MolFromSmiles, smiles_list))
        assert None not in mols, "Invalid SMILES encountered."
        self._fps = list(map(self.get_fingerprint, mols))

    def _get_nearest_neighbour_dist(self, smiles: str) -> float:
        fp_query = self.get_fingerprint(AllChem.MolFromSmiles(smiles))
        dice_sims = DataStructs.BulkDiceSimilarity(fp_query, self._fps)
        return 1 - max(dice_sims)

    def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
        if len(nodes) == 0:
            return []

        # Get distances to nearest neighbours
        nn_dists = np.asarray(
            [self._get_nearest_neighbour_dist(node.mol.smiles) for node in nodes]
        )
        assert np.min(nn_dists) >= 0

        # Turn into costs
        if self.distance_to_cost == DistanceToCost.NOTHING:
            values = nn_dists
        elif self.distance_to_cost == DistanceToCost.EXP:
            values = np.exp(nn_dists) - 1
        elif self.distance_to_cost == DistanceToCost.QUAD:
            values = np.power(nn_dists,2)
        elif self.distance_to_cost == DistanceToCost.CUB:
            values = np.power(nn_dists,3)
        elif self.distance_to_cost == DistanceToCost.SIN:
            values = np.sin(nn_dists)
        elif self.distance_to_cost == DistanceToCost.LOG:
            values = np.log(1 + nn_dists)
        else:
            raise NotImplementedError(self.distance_to_cost)

        return list(values)



##############
# DOESN'T WORK
###############
# class CosineNNCostEstimator(NoCacheNodeEvaluator):
#     '''Estimate the cost of Nearest neighbor using Dice distance function'''

#     def __init__(
#         self,
#         inventory: ExplicitMolInventory,
#         distance_to_cost: DistanceToCost,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.distance_to_cost = distance_to_cost
#         self._set_fingerprints([mol.smiles for mol in inventory.purchasable_mols()])

#     def get_fingerprint(self, mol: AllChem.Mol):
#         return AllChem.GetMorganFingerprint(mol, radius=3)

#     def _set_fingerprints(self, smiles_list: list[str]) -> None:
#         """Initialize fingerprint cache."""
#         mols = list(map(AllChem.MolFromSmiles, smiles_list))
#         assert None not in mols, "Invalid SMILES encountered."
#         self._fps = list(map(self.get_fingerprint, mols))

#     def _get_nearest_neighbour_dist(self, smiles: str) -> float:
#         fp_query = self.get_fingerprint(AllChem.MolFromSmiles(smiles))
#         dice_sims = DataStructs.BulkCosineSimilarity(fp_query, self._fps)
#         return 1 - max(dice_sims)

#     def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
#         if len(nodes) == 0:
#             return []

#         # Get distances to nearest neighbours
#         nn_dists = np.asarray(
#             [self._get_nearest_neighbour_dist(node.mol.smiles) for node in nodes]
#         )
#         assert np.min(nn_dists) >= 0

#         # Turn into costs
#         if self.distance_to_cost == DistanceToCost.NOTHING:
#             values = nn_dists
#         elif self.distance_to_cost == DistanceToCost.EXP:
#             values = np.exp(nn_dists) - 1
#         elif self.distance_to_cost == DistanceToCost.QUAD:
#             values = np.power(nn_dists,2)
#         elif self.distance_to_cost == DistanceToCost.CUB:
#             values = np.power(nn_dists,3)
#         elif self.distance_to_cost == DistanceToCost.SIN:
#             values = np.sin(nn_dists)
#         elif self.distance_to_cost == DistanceToCost.LOG:
#             values = np.log(1 + nn_dists)
#         else:
#             raise NotImplementedError(self.distance_to_cost)

#         return list(values)
    

class CustomNNCostEstimator(NoCacheNodeEvaluator):
    '''Estimate the cost of Nearest neighbor using custom distance function'''

    def __init__(
        self,
        inventory: ExplicitMolInventory,
        distance_to_cost: DistanceToCost,
        similarity_metric,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.distance_to_cost = distance_to_cost
        self._set_fingerprints([mol.smiles for mol in inventory.purchasable_mols()])
        self.similarity_metric = similarity_metric

    def get_fingerprint(self, mol: AllChem.Mol):
        return AllChem.GetMorganFingerprint(mol, radius=3)

    def _set_fingerprints(self, smiles_list: list[str]) -> None:
        """Initialize fingerprint cache."""
        mols = list(map(AllChem.MolFromSmiles, smiles_list))
        assert None not in mols, "Invalid SMILES encountered."
        self._fps = list(map(self.get_fingerprint, mols))

    def _get_nearest_neighbour_dist(self, smiles: str) -> float:
        fp_query = self.get_fingerprint(AllChem.MolFromSmiles(smiles))
        dice_sims = self.similarity_metric(fp_query, self._fps)
        return 1 - max(dice_sims)

    def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
        if len(nodes) == 0:
            return []

        # Get distances to nearest neighbours
        nn_dists = np.asarray(
            [self._get_nearest_neighbour_dist(node.mol.smiles) for node in nodes]
        )
        assert np.min(nn_dists) >= 0

        # Turn into costs
        if self.distance_to_cost == DistanceToCost.NOTHING:
            values = nn_dists
        elif self.distance_to_cost == DistanceToCost.EXP:
            values = np.exp(nn_dists) - 1
        elif self.distance_to_cost == DistanceToCost.QUAD:
            values = np.power(nn_dists,2)
        elif self.distance_to_cost == DistanceToCost.CUB:
            values = np.power(nn_dists,3)
        elif self.distance_to_cost == DistanceToCost.SIN:
            values = np.sin(nn_dists)
        elif self.distance_to_cost == DistanceToCost.LOG:
            values = np.log(1 + nn_dists)
        else:
            raise NotImplementedError(self.distance_to_cost)

        return list(values)

'''
similarityFunctions = [
  ('Tanimoto', TanimotoSimilarity, ''),
  ("Dice", DiceSimilarity, ''),
  ("Cosine", CosineSimilarity, ''),
  ("Sokal", SokalSimilarity, ''),
  ("Russel", RusselSimilarity, ''),
  ("RogotGoldberg", RogotGoldbergSimilarity, ''),
  ("AllBit", AllBitSimilarity, ''),
  ("Kulczynski", KulczynskiSimilarity, ''),
  ("McConnaughey", McConnaugheySimilarity, ''),
  ("Asymmetric", AsymmetricSimilarity, ''),
  ("BraunBlanquet", BraunBlanquetSimilarity, ''),
]
'''


# Correlation between distance and true rollout value (cost) -> min depth of the tree

# Read the Retro * paper for the cost function

# More iterations