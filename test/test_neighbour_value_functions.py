import numpy as np

from syntheseus.search.chem import Molecule
from syntheseus.search.graph.and_or import OrNode
from syntheseus.search.mol_inventory import SmilesListInventory
from neighbour_value_functions import TanimotoNNCostEstimator, DistanceToCost


def test_tanimoto_nn_cost_estimator() -> None:
    # Set up inventory
    inventory = SmilesListInventory(["C", "CC[OH]", "O=O"])
    cost_fn = TanimotoNNCostEstimator(
        inventory=inventory, distance_to_cost=DistanceToCost.NOTHING
    )

    # Test values with no cost function
    test_smiles = [
        "C",  # in the inventory
        "c1ccccc1",  # no shared components in inventory (Tanimoto distance = 1)
        "CC(=O)C",  # shares bits with all molecules in the inventory
    ]
    expected_nn_distances = [0.0, 1.0, 0.9]
    nodes = [OrNode(Molecule(s)) for s in test_smiles]
    costs = cost_fn(nodes)
    assert np.allclose(np.asarray(costs), np.asarray(expected_nn_distances))

    # Second try using EXP modifier
    cost_fn.distance_to_cost = DistanceToCost.EXP
    costs = cost_fn(nodes)
    assert np.allclose(np.asarray(costs), np.exp(np.asarray(expected_nn_distances)) - 1)
