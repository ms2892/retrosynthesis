from __future__ import annotations

import argparse
import logging
import sys
import numpy as np

from tqdm.auto import tqdm
from collections import defaultdict
from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit import DataStructs
from syntheseus.search.chem import Molecule
from syntheseus.search.graph.and_or import AndNode
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.analysis.route_extraction import min_cost_routes
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode
from syntheseus.search.analysis import route_extraction
from syntheseus.search.reaction_models.base import BackwardReactionModel
from syntheseus.search.mol_inventory import BaseMolInventory
from syntheseus.search.node_evaluation.base import (
    BaseNodeEvaluator,
    NoCacheNodeEvaluator,
)
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator

from paroutes import PaRoutesInventory, PaRoutesModel, get_target_smiles
from rdkit.Chem import DataStructs, AllChem
from neighbour_value_functions import TanimotoNNCostEstimator, DistanceToCost, DiceNNCostEstimator, CustomNNCostEstimator

class PaRoutesRxnCost(NoCacheNodeEvaluator[AndNode]):
    """Cost of reaction is negative log softmax, floored at -3."""

    def _evaluate_nodes(self, nodes: list[AndNode], graph=None) -> list[float]:
        softmaxes = np.asarray([node.reaction.metadata["softmax"] for node in nodes])
        costs = np.clip(-np.log(softmaxes), 1e-1, 10.0)
        return costs.tolist()

def calc_tanimoto(smile1,smile2):
    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1,3,nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2,3,nBits=2048)

    s = round(DataStructs.TanimotoSimilarity(fp1,fp2),3)
    return s

def calculate_correlation(
    smiles_list: list[str],
    value_functions: list[tuple[str, BaseNodeEvaluator]],
    rxn_model: BackwardReactionModel,
    inventory: BaseMolInventory,
    use_tqdm: bool = False,
    limit_rxn_model_calls: int = 1000,
    limit_iterations: int = 1_000_000,
) -> list[tuple[float, ...]]:
    """
    Do search on a list of SMILES strings and report the time of first solution.
    """

    # Initialize algorithm.
    common_kwargs = dict(
        reaction_model=rxn_model,
        mol_inventory=inventory,
        limit_reaction_model_calls=limit_rxn_model_calls,
        limit_iterations=limit_iterations,
        max_expansion_depth=15,  # prevent overly-deep solutions
        prevent_repeat_mol_in_trees=True,  # original paper did this
    )
    algs = [
        RetroStarSearch(
            and_node_cost_fn=PaRoutesRxnCost(), value_function=fn, **common_kwargs
        )
        for _, fn in value_functions
    ]

    # Do search
    logger = logging.getLogger("COMPARISON")
    min_soln_times: list[tuple[float, ...]] = []
    if use_tqdm:
        smiles_iter = tqdm(smiles_list)
    else:
        smiles_iter = smiles_list
    points = []
    for i, smiles in enumerate(smiles_iter):
        logger.debug(f"Start search {i}/{len(smiles_list)}. SMILES: {smiles}")
        this_soln_times = list()
        for (name, _), alg in zip(value_functions, algs):
            alg.reset()
            output_graph, _ = alg.run_from_mol(Molecule(smiles))
            for nodes in route_extraction.min_cost_routes(output_graph,max_routes=100):
                subgraph = output_graph._graph.subgraph(nodes)
                edge_list = defaultdict(list)
                lefts=set()
                rights=set()
                for node in subgraph.nodes:
                    for child in subgraph.successors(node):
                        lefts.add(node)
                        rights.add(child)
                        edge_list[node].append(child)
                root = lefts.difference(rights)
                root = list(root)
                root = root[0]
                dp_table={}
                for node in subgraph.nodes:
                    if isinstance(node,OrNode):
                        if node.mol.metadata['is_purchasable']:
                            dp_table[node]=calc_tanimoto(node.mol.smiles,root.mol.smiles)

                mxm_depth = -1

                # Change to BFS
                def dfs_depth(root,curr_depth):
                    nonlocal mxm_depth
                    if isinstance(root,OrNode):
                        if root.mol.metadata['is_purchasable']:
                            mxm_depth = max(mxm_depth,curr_depth)
                            return
                        else:
                            for child in subgraph.successors(root):
                                dfs_depth(child,curr_depth+1)
                    else:
                        for child in subgraph.successors(root):
                            dfs_depth(child,curr_depth+1)
                dfs_depth(root,0)
                print(mxm_depth)
                x=input()
                for purchasable in dp_table.keys():
                    points.append((mxm_depth,dp_table[purchasable])) 
                print(points)
    return points
            # Graph has been made adjacency list with root as well in edges and root variable respectively. tanimoto distances are in dp_table

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit_num_smiles",
        type=int,
        default=None,
        help="Maximum number of SMILES to run.",
    )
    parser.add_argument(
        "--limit_iterations",
        type=int,
        default=100000,
        help="Maximum number of algorithm iterations.",
    )
    parser.add_argument(
        "--limit_rxn_model_calls",
        type=int,
        default=500,
        help="Allowed number of calls to reaction model.",
    )
    parser.add_argument(
        "--paroutes_n",
        type=int,
        default=5,
        help="Which PaRoutes benchmark to use.",
    )
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        filemode="w",
    )
    logging.getLogger().info(args)

    # Load all SMILES to test
    test_smiles = get_target_smiles(args.paroutes_n)
    if args.limit_num_smiles is not None:
        test_smiles = test_smiles[: args.limit_num_smiles]

    # Make reaction model, inventory, value functions
    rxn_model = PaRoutesModel()
    inventory = PaRoutesInventory(n=args.paroutes_n)
    value_fns = [
        (
            "Tanimoto-distance-EXP",
            TanimotoNNCostEstimator(
                inventory=inventory, distance_to_cost=DistanceToCost.EXP
            ),
        )
    ]

    # Run without value function (retro*-0)
    points = calculate_correlation(
        smiles_list=test_smiles[:1000],
        value_functions=value_fns,
        limit_rxn_model_calls=args.limit_rxn_model_calls,
        limit_iterations=args.limit_iterations,
        use_tqdm=True,
        rxn_model=rxn_model,
        inventory=inventory,
    )

    points = np.array(points)

    np.save('points.npy',points)
    