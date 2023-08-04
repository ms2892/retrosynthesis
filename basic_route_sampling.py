from __future__ import annotations
import random
from typing import Callable, Optional

from syntheseus.search.graph.and_or import AndOrGraph, AndNode, OrNode, ANDOR_NODE


def _uniform_solved_weight(node: AndNode) -> float:
    return float(node.has_solution)


def sample_synthesis_route(
    graph: AndOrGraph,
    and_node_weight_function: Callable[[AndNode], float] = _uniform_solved_weight,
) -> Optional[ANDOR_NODE]:
    """
    Randomly sample a synthesis route from a graph.

    Starting from the root node, randomly choose an AndNode child at each step
    with probability proportional to its weight from `and_node_weight_function`.
    Then, sample a route from each OrNode child.

    For a tree, the result will always be a randomly chosen route.
    For a graph, in some cases the sampling may fail due to loops
    (e.g. sampling A -> B -> A).
    If this happens, it will return None.

    NOTE: at the moment this method is hard-coded to stop at leaf nodes,
    but this could change in the future.
    """

    return _recursive_sample_synthesis_route(
        node=graph.root_node,
        graph=graph,
        and_node_weight_function=and_node_weight_function,
        nodes_previously_visited=set(),
    )


def _recursive_sample_synthesis_route(
    node: OrNode,
    graph: AndOrGraph,
    and_node_weight_function: Callable[[AndNode], float],
    nodes_previously_visited: set[ANDOR_NODE],
) -> Optional[set[ANDOR_NODE]]:
    """Main function for sampling a synthesis route."""

    # Base case 1: node is already seen. Return None
    if node in nodes_previously_visited:
        return None

    # Base case 2: node has no children, so stop sampling here
    children = list(graph.successors(node))
    nodes_visited_so_far = nodes_previously_visited | {node}
    if len(children) == 0:
        return nodes_visited_so_far

    # Main case: sample a child AndNode, then visit samples
    children_weights = [and_node_weight_function(n) for n in children]
    assert all(w >= 0 for w in children_weights), "weights must be non-negative"
    assert any(w > 0 for w in children_weights), "at least one weight must be positive"
    chosen_and_child = random.choices(children, weights=children_weights, k=1)[0]
    nodes_visited_so_far.add(chosen_and_child)
    for or_node_grandchild in graph.successors(chosen_and_child):
        # Recursively sample the route from this child
        nodes_visited_for_grandchild = _recursive_sample_synthesis_route(
            node=or_node_grandchild,
            graph=graph,
            and_node_weight_function=and_node_weight_function,
            nodes_previously_visited=nodes_visited_so_far,
        )

        if nodes_visited_for_grandchild is None:
            # If the sample failed, return None
            return None
        else:
            # Update nodes visited so far and continue sampling
            nodes_visited_so_far |= nodes_visited_for_grandchild

    # If this point is reached, then all samples succeeded.
    # return the nodes sampled
    return nodes_visited_so_far
