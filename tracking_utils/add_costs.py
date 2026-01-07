from typing import Dict, List, Tuple

from tracking_utils.library_costs import EdgeSelection
from motile.costs import Appear, Disappear
from motile.solver import Solver


def add_costs(
    solver: Solver,
    attributes: List[str],
    statistics: Dict[str, Tuple[float, float]],
) -> None:
    """Modify solver in place by adding edge distance, appear, and disappear costs."""
    for attr in attributes:
        solver.add_cost(
            EdgeSelection(
                weight=1.0, constant=0.0, attribute=attr, statistics=statistics[attr]
            ),
            name=f"Edge Selection {attr}",
        )

    solver.add_cost(
        Appear(weight=0.0, constant=1.0, ignore_attribute="ignore_appear_cost")
    )
    solver.add_cost(Disappear(constant=1.0, ignore_attribute="ignore_disappear_cost"))
