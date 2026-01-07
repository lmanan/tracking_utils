from motile.constraints import MaxChildren, MaxParents
from motile.solver import Solver
from tracking_utils.library_constraints import Cardinality

def add_constraints(
    solver: Solver,
    max_children: int = 2,
    num_tracks: int | None = None,

) -> None:
    """Modify solver in place by adding tracking constraints.

    Args:
        solver: The motile Solver instance to modify.
        max_children: Maximum number of children per node (1 or 2). Defaults to 2.
        num_tracks: Exact number of tracks to find. If None, no cardinality constraint is added.
    """
    solver.add_constraint(MaxParents(1))
    if max_children == 1:
        solver.add_constraint(MaxChildren(1))
    elif max_children == 2:
        solver.add_constraint(MaxChildren(2))
    if num_tracks is not None:
        solver.add_constraint(Cardinality(num_tracks))
