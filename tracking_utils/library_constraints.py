from collections import defaultdict

from motile.constraints import Constraint
from motile.variables import NodeAppear, NodeSelected


class ExactTrackCount(Constraint):
    """Constraint that enforces exactly num_tracks tracks in the entire data.

    A track is counted by its appearance (NodeAppear) indicator. Summing all
    NodeAppear indicators across every node in the graph and requiring the sum
    to equal num_tracks ensures that precisely that many tracks are present in
    the full solution.
    """

    def __init__(self, num_tracks):
        self.num_tracks = num_tracks

    def instantiate(self, solver):
        appear_indicators = solver.get_variables(NodeAppear)
        yield sum([appear_indicators[n] for n in solver.graph.nodes]) == self.num_tracks


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
