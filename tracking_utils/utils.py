from typing import Tuple


def is_hyper_edge(edge: Tuple) -> bool:
    """Check if an edge is a hyper-edge.

    Matches motile's TrackGraph.is_hyperedge() logic:
    - Regular edge: (u, v) where both are node IDs
    - Hyper-edge: ((nodes...), (nodes...)) where both are tuples
    """
    src, target = edge
    return isinstance(src, tuple) and isinstance(target, tuple)
