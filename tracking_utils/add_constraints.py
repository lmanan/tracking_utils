from typing import Optional

from motile.constraints import MaxChildren, MaxParents, Pin
from motile.solver import Solver
from tracking_utils.library_constraints import (
    ExactTrackCount,
    ExactSelectionsPerFrame,
    ExactActiveTrackletsPerFrame,
)


def add_constraints(
    solver: Solver,
    max_children: int = 2,
    num_tracks: Optional[int] = None,
    set_exact_track_count: bool = False,
    set_exact_selections_per_frame: bool = False,
    set_exact_active_tracklets_per_frame: bool = False,
    pin_attribute: Optional[str] = None,
) -> None:
    """Modify solver in place by adding tracking constraints.

    Args:
        solver: The motile Solver instance to modify.
        max_children: Maximum number of children per node (1 or 2). Defaults to 2.
        num_tracks: Exact number of tracks to find. Required when any cardinality
            constraint is enabled.
        set_exact_track_count: If True, adds a global track count constraint enforcing
            that the total number of tracks equals num_tracks. Default is False.
        set_exact_selections_per_frame: If True, adds a per-frame constraint enforcing
            that exactly num_tracks nodes are selected in every frame. Default is False.
        set_exact_active_tracklets_per_frame: If True, adds a per-frame constraint
            enforcing that exactly num_tracks tracklets are active (t_start <= t <= t_end)
            in every frame. Intended for tracklet stitching. Default is False.
        pin_attribute: Node attribute name to use for pinning. Nodes with this attribute
            set to True will be forced to be selected in the solution. If None, no
            pin constraint is added. Common usage: pin_attribute="GT" for ground truth.
    """
    solver.add_constraint(MaxParents(1))
    if max_children == 1:
        solver.add_constraint(MaxChildren(1))
    elif max_children == 2:
        solver.add_constraint(MaxChildren(2))
    if set_exact_track_count:
        solver.add_constraint(ExactTrackCount(num_tracks))
    if set_exact_selections_per_frame:
        solver.add_constraint(ExactSelectionsPerFrame(num_tracks))
    if set_exact_active_tracklets_per_frame:
        solver.add_constraint(ExactActiveTrackletsPerFrame(num_tracks))
    if pin_attribute is not None:
        solver.add_constraint(Pin(attribute=pin_attribute))
