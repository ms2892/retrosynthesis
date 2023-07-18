"""Script to compare heuristic value functions using the Paroutes inventory / environment."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence
import numpy as np

from neighbour_value_functions import TanimotoNNCostEstimator, DistanceToCost
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.search.graph.and_or import AndOrGraph, OrNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator

from paroutes import PaRoutesInventory, PaRoutesModel
from example_paroutes import compare_cost_functions
from heuristic_value_functions import *


class FiniteMolIsPurchasableCost(NoCacheNodeEvaluator[OrNode]):
    def __init__(self, non_purchasable_cost: float, **kwargs):
        super().__init__(**kwargs)
        self.non_purchasable_cost = non_purchasable_cost

    def _evaluate_nodes(  # type: ignore[override]
        self,
        nodes: Sequence[OrNode],
        graph: Optional[AndOrGraph] = None,
    ) -> list[float]:
        return [
            0.0
            if node.mol.metadata.get("is_purchasable")
            else self.non_purchasable_cost
            for node in nodes
        ]


class ScaledTanimotoNNAvgCostEstimator(TanimotoNNCostEstimator):
    """
    Value function which is:

    v(mol) = scale * distance_to_cost(average(d(mol, m'))) over top N nearest neighbours m'
    """

    def __init__(self, scale: float, num_nearest_neighbours: int = 1, **kwargs):
        super().__init__(num_top_sims_to_return=num_nearest_neighbours, **kwargs)
        self.scale = scale

    def _evaluate_nodes(self, nodes: list, graph=None) -> list[float]:
        if len(nodes) == 0:
            return []

        # Get distances to nearest neighbours
        nn_dists = np.asarray(
            [self._get_nearest_neighbour_dists(node.mol.smiles) for node in nodes]
        )
        assert np.min(nn_dists) >= 0  # ensure distances are valid

        # Average over topN distances
        avg_topn_dists = np.mean(nn_dists, axis=1)

        # Turn into costs
        if self.distance_to_cost == DistanceToCost.NOTHING:
            non_linear_costs = avg_topn_dists
        elif self.distance_to_cost == DistanceToCost.EXP:
            non_linear_costs = np.exp(avg_topn_dists) - 1
        else:
            raise NotImplementedError(self.distance_to_cost)

        # Scale and return
        return (self.scale * non_linear_costs).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--limit_rxn_model_calls",
        type=int,
        default=25,
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
    with open(args.smiles_file, "r") as f:
        test_smiles = [line.strip() for line in f.readlines()]

    # Make reaction model, inventory, value functions
    rxn_model = PaRoutesModel()
    inventory = PaRoutesInventory(n=args.paroutes_n)

    # Create all the value functions to test
    value_fns = [  # baseline: 0 value function
        ("constant-0", ConstantNodeEvaluator(0.0)),
    ]

    # Nearest neighbour cost heuristics
    # for num_nearest_neighbours in [
    #     1,
    #     5,
    # ]:
    #     for scale in [
    #         1.0,
    #         3.0,
    #     ]:
    #         # Tanimoto distance cost heuristic
    #         value_fns.append(
    #             (
    #                 f"Tanimoto-top{num_nearest_neighbours}NN-linear-{scale}",
    #                 ScaledTanimotoNNAvgCostEstimator(
    #                     scale=scale,
    #                     inventory=inventory,
    #                     distance_to_cost=DistanceToCost.NOTHING,
    #                     num_nearest_neighbours=num_nearest_neighbours,
    #                 ),
    #             )
    #         )

    for scale in [0.3, 1.0]:
        # SAscore cost heuristic (different scale)
        value_fns.append(
            (
                f"SAscore-linear-{scale}",
                ScaledSAScoreCostFunction(
                    scale=scale,
                ),
            )
        )

    for scale in [0.3, 1.0]:
        # SAscore cost heuristic (different scale)
        value_fns.append(
            (
                f"GNNEstimatorMAXScore-{scale}",
                GNNEstimatorFunction(
                    scale=scale,
                    inventory=inventory,
                    mode='max'
                ),
            )
        )

    for scale in [0.3, 1.0]:
        # SAscore cost heuristic (different scale)
        value_fns.append(
            (
                f"GNNEstimatorAVGScore-{scale}",
                GNNEstimatorFunction(
                    scale=scale,
                    inventory=inventory,
                    mode='avg'
                ),
            )
        )

    # Run with all value functions
    soln_times_list = compare_cost_functions(
        smiles_list=test_smiles,
        value_functions=value_fns,
        limit_rxn_model_calls=args.limit_rxn_model_calls,
        limit_iterations=10_000_000,
        use_tqdm=True,
        rxn_model=rxn_model,
        inventory=inventory,
        rxn_cost_fn=ConstantNodeEvaluator(1.0),
        stop_on_first_solution=True,  # we only track the first solution time
        unique_nodes=True,  # save memory
        prevent_repeat_mol_in_trees=False,  # since unique_nodes=True
        max_routes_to_extract=0,  # prevent memory issues in route extraction
        or_node_cost_fn=FiniteMolIsPurchasableCost(non_purchasable_cost=1e4),
    )

    # Print quantiles of solution time
    soln_times_array = np.asarray(soln_times_list)
    print("Solution time quantiles:\n")
    for i, (name, _) in enumerate(value_fns):
        print(f"Value function: {name}")
        for q in [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
            print(f"Quantile {q}: {np.quantile(soln_times_array[:, i], q):.2f}")
        print()
