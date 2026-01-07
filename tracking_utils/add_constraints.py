from motile.constraints import MaxChildren, MaxParents
from motile.solver import Solver


def add_constraints(
    solver: Solver,
    max_children: int = 2,
) -> None:
    """Modify solver in place by adding tracking constraints.

    Args:
        solver: The motile Solver instance to modify.
        max_children: Maximum number of children per node (1 or 2). Defaults to 2.
    """
    solver.add_constraint(MaxParents(1))
    if max_children == 1:
        solver.add_constraint(MaxChildren(1))
    elif max_children == 2:
        solver.add_constraint(MaxChildren(2))
