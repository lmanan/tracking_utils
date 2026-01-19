from typing import Dict, List, Tuple, Union

from tracking_utils.library_costs import EdgeSelection
from motile.costs import Appear, Disappear
from motile.solver import Solver


def add_costs(
    solver: Solver,
    attributes: List[Union[str, Tuple[str, ...]]],
    statistics: Dict[str, Dict[str, Tuple[float, float]]],
) -> None:
    """Modify solver in place by adding edge distance, appear, and disappear costs.

    Args:
        solver: The motile Solver instance.
        attributes: List of attribute names (str or tuple of str) to add costs for.
        statistics: Dictionary mapping attribute names to a dict with 'regular' and
            'hyper' keys, each containing (mean, std) tuples.
    """
    for attr in attributes:
        attr_key = attr if isinstance(attr, str) else attr[0]
        solver.add_cost(
            EdgeSelection(
                attribute=attr,
                regular_weight=1.0,
                regular_constant=0.0,
                hyper_weight=1.0,
                hyper_constant=0.0,
                regular_statistics=statistics[attr_key]["regular"],
                hyper_statistics=statistics[attr_key]["hyper"],
            ),
            name=f"Edge Selection {attr_key}",
        )

    solver.add_cost(
        Appear(weight=0.0, constant=1.0, ignore_attribute="ignore_appear_cost")
    )
    solver.add_cost(Disappear(constant=1.0, ignore_attribute="ignore_disappear_cost"))
