"""
Demo script comparing nearest neighbour cost function with constant value function on PaRoutes.
"""
from __future__ import annotations

import argparse
import logging
import sys
import numpy as np

from tqdm.auto import tqdm

from syntheseus.search.chem import Molecule
from syntheseus.search.graph.and_or import AndNode
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.analysis.route_extraction import iter_routes_cost_order
from syntheseus.search.reaction_models.base import BackwardReactionModel
from syntheseus.search.mol_inventory import BaseMolInventory
from syntheseus.search.node_evaluation.base import (
    BaseNodeEvaluator,
    NoCacheNodeEvaluator,
)
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator

from paroutes import PaRoutesInventory, PaRoutesModel, get_target_smiles
from neighbour_value_functions import TanimotoNNCostEstimator, DistanceToCost
from faster_retro_star import ReduceValueFunctionCallsRetroStar


class PaRoutesRxnCost(NoCacheNodeEvaluator[AndNode]):
    """Cost of reaction is negative log softmax, floored at -3."""

    def _evaluate_nodes(self, nodes: list[AndNode], graph=None) -> list[float]:
        softmaxes = np.asarray([node.reaction.metadata["softmax"] for node in nodes])
        costs = np.clip(-np.log(softmaxes), 1e-1, 10.0)
        return costs.tolist()


def compare_cost_functions(
    smiles_list: list[str],
    value_functions: list[tuple[str, BaseNodeEvaluator]],
    rxn_model: BackwardReactionModel,
    inventory: BaseMolInventory,
    rxn_cost_fn: BaseNodeEvaluator,
    use_tqdm: bool = False,
    limit_rxn_model_calls: int = 100,
    limit_iterations: int = 1_000_000,
    prevent_repeat_mol_in_trees: bool = True,  # original paper did this
    max_routes_to_extract: int = 10,
    **alg_kwargs,
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
        max_expansion_depth=30,  # prevent overly-deep solutions
        prevent_repeat_mol_in_trees=prevent_repeat_mol_in_trees,  # original paper did this
        **alg_kwargs,
    )
    algs = [
        ReduceValueFunctionCallsRetroStar(
            and_node_cost_fn=rxn_cost_fn, value_function=fn, **common_kwargs
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
    for i, smiles in enumerate(smiles_iter):
        logger.debug(f"Start search {i}/{len(smiles_list)}. SMILES: {smiles}")
        this_soln_times = list()
        for (name, _), alg in zip(value_functions, algs):
            alg.reset()
            output_graph, _ = alg.run_from_mol(Molecule(smiles))

            # Analyze solution time
            for node in output_graph.nodes():
                node.data["analysis_time"] = node.data["num_calls_rxn_model"]
            soln_time = get_first_solution_time(output_graph)
            this_soln_times.append(soln_time)

            # Analyze number of routes
            routes = list(iter_routes_cost_order(output_graph, max_routes_to_extract))

            # Print result
            if alg.reaction_model.num_calls() < limit_rxn_model_calls:
                note = " (NOTE: this was less than the maximum budget)"
            else:
                note = ""
            logger.debug(
                f"Done {name}: nodes={len(output_graph)}, solution time = {soln_time}, "
                f"num routes = {len(routes)} (capped at {max_routes_to_extract}), "
                f"final num rxn model calls = {alg.reaction_model.num_calls()}{note}, "
                f"final num value model calls = {alg.value_function.num_calls}."
            )
        min_soln_times.append(tuple(this_soln_times))

    return min_soln_times


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
        default=500,
        help="Maximum number of algorithm iterations.",
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
    parser.add_argument(
        "--rxn_cost",
        type=str,
        default="log-softmax",
        help="Cost function to use.",
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
        ("constant-0", ConstantNodeEvaluator(0.0)),
        (
            "Tanimoto-distance",
            TanimotoNNCostEstimator(
                inventory=inventory, distance_to_cost=DistanceToCost.NOTHING
            ),
        ),
    ]
    if args.rxn_cost == "log-softmax":
        rxn_cost_fn = PaRoutesRxnCost()
    elif args.rxn_cost == "constant":
        rxn_cost_fn = ConstantNodeEvaluator(1.0)
    else:
        raise ValueError(f"Unknown rxn_cost: {args.rxn_cost}")

    # Run with all value functions
    soln_times_list = compare_cost_functions(
        smiles_list=test_smiles,
        value_functions=value_fns,
        limit_rxn_model_calls=args.limit_rxn_model_calls,
        limit_iterations=args.limit_iterations,
        use_tqdm=True,
        rxn_model=rxn_model,
        inventory=inventory,
        rxn_cost_fn=rxn_cost_fn,
    )

    # Print quantiles of solution time
    soln_times_array = np.asarray(soln_times_list)
    for i, (name, _) in enumerate(value_fns):
        for q in [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
            print(f"{name} quantile {q}: {np.quantile(soln_times_array[:, i], q):.2f}")
