from typing import Dict, List, Tuple, Union

from tracking_utils.library_costs import EdgeSelection, NodeSelection
from motile.costs import Appear, Disappear, Split
from motile.solver import Solver


def add_costs(
    solver: Solver,
    edge_attributes: List[Union[str, Tuple[str, ...]]],
    edge_statistics: Dict[str, Dict[str, Tuple[float, float]]],
    node_attributes: List[str] = None,
    node_statistics: Dict[str, Tuple[float, float]] = None,
    use_split_cost: bool = False,
    add_hyper_edges: bool = False,
) -> None:
    """Modify solver in place by adding edge distance, appear, and disappear costs.

    Args:
        solver: The motile Solver instance.
        edge_attributes: List of attribute names (str or tuple of str) to add edge costs for.
        edge_statistics: Dictionary mapping attribute names to a dict with 'regular' and
            'hyper' keys, each containing (mean, std) tuples for edge statistics.
        node_attributes: Optional list of node attribute names to add node selection
            costs for. Nodes with low attribute values are encouraged to be selected.
        node_statistics: Optional dictionary mapping node attribute names to (mean, std)
            tuples for z-score normalization of node costs.
    """
    # Add node selection costs
    if node_attributes:
        node_statistics = node_statistics or {}
        for attr in node_attributes:
            solver.add_cost(
                NodeSelection(
                    attribute=attr,
                    weight=1.0,
                    constant=0.0,
                    statistics=node_statistics.get(attr),
                ),
                name=f"Node Selection {attr}",
            )

    # Add edge selection costs
    for attr in edge_attributes:
        attr_key = attr if isinstance(attr, str) else attr[0]
        solver.add_cost(
            EdgeSelection(
                attribute=attr,
                regular_weight=1.0,
                regular_constant=0.0,
                hyper_weight=1.0,
                hyper_constant=0.0,
                regular_statistics=edge_statistics[attr_key]["regular"],
                hyper_statistics=edge_statistics[attr_key]["hyper"],
                use_hyper_edges=add_hyper_edges,
            ),
            name=f"Edge Selection {attr_key}",
        )

    if use_split_cost and not add_hyper_edges:
        solver.add_cost(Split(weight=0.0, constant=1.0))

    solver.add_cost(
        Appear(weight=0.0, constant=1.0, ignore_attribute="ignore_appear_cost")
    )
    solver.add_cost(Disappear(constant=1.0, ignore_attribute="ignore_disappear_cost"))
