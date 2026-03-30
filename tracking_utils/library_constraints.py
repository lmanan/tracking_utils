from collections import defaultdict

from motile.constraints import Constraint
from motile.variables import EdgeSelected, NodeSelected


class ExactTrackCount(Constraint):
    """Constraint that enforces exactly num_tracks tracks in the entire data.

    A track start is defined as a selected node with no selected incoming edge.
    For each node this equals NodeSelected[v] - sum(EdgeSelected[e] for e in
    in_edges(v)), which is 1 for a genuine chain start and 0 otherwise.

    Note: NodeAppear was intentionally not used here. motile's NodeAppear
    defines prev_edges using frame-adjacency (nodes at time t-1), not graph
    topology. In tracklet stitching, edges can span many frames (e.g. a
    tracklet ending at t=61 connecting to one starting at t=62 is placed at
    t_start=0 and t_start=62 respectively). motile therefore sees no
    frame-adjacent predecessor for the target node, falls into the
    no-incoming-edges branch (appear == selected), and marks it as a chain
    start even when a stitching edge is selected. This phantom appear lets the
    solver satisfy the count constraint while producing fewer real chains than
    required.
    """

    def __init__(self, num_tracks):
        self.num_tracks = num_tracks

    def instantiate(self, solver):
        node_indicators = solver.get_variables(NodeSelected)
        edge_indicators = solver.get_variables(EdgeSelected)

        chain_starts = []
        for node in solver.graph.nodes:
            in_edges = solver.graph.prev_edges[node]
            selected = node_indicators[node]
            if not in_edges:
                chain_starts.append(selected)
            else:
                incoming = sum(edge_indicators[e] for e in in_edges)
                chain_starts.append(selected - incoming)

        yield sum(chain_starts) == self.num_tracks


class ExactSelectionsPerFrame(Constraint):
    """Constraint that enforces exactly num_tracks nodes with NodeSelected set to 1 per frame.

    For each time frame, the sum of NodeSelected indicators over all nodes in
    that frame is required to equal num_tracks. This guarantees that exactly
    that many detections are selected in every frame of the solution.
    """

    def __init__(self, num_tracks):
        self.num_tracks = num_tracks

    def instantiate(self, solver):
        node_indicators = solver.get_variables(NodeSelected)
        nodes_by_frame = defaultdict(list)
        for node in solver.graph.nodes:
            t = solver.graph.nodes[node].get("time")
            if t is not None:
                nodes_by_frame[t].append(node)
        for nodes in nodes_by_frame.values():
            yield sum([node_indicators[n] for n in nodes]) == self.num_tracks


class ExactActiveTrackletsPerFrame(Constraint):
    """Constraint that enforces exactly num_tracks tracklets active in every frame.

    A tracklet node is considered active at frame t if t_start <= t <= t_end.
    For each frame, the sum of NodeSelected indicators over all tracklets active
    at that frame is required to equal num_tracks.

    Unlike ExactSelectionsPerFrame, each tracklet contributes to the sum of
    every frame it spans, not just the frame it starts at.
    """

    def __init__(self, num_tracks: int):
        self.num_tracks = num_tracks

    def instantiate(self, solver):
        node_indicators = solver.get_variables(NodeSelected)

        # Collect all frames spanned by any tracklet
        nodes_by_frame = defaultdict(list)
        for node in solver.graph.nodes:
            attrs = solver.graph.nodes[node]
            t_start = attrs.get("t_start")
            t_end = attrs.get("t_end")
            if t_start is None or t_end is None:
                continue
            for t in range(int(t_start), int(t_end) + 1):
                nodes_by_frame[t].append(node)

        for nodes in nodes_by_frame.values():
            yield sum([node_indicators[n] for n in nodes]) == self.num_tracks
