import logging
import numpy as np
from motile.costs import Cost, Weight
from motile.solver import Solver
from typing import Optional, Tuple, Union
from motile.variables import EdgeSelected, NodeSelected

from tracking_utils.utils import is_hyper_edge

logger = logging.getLogger(__name__)


def _unwrap_node_id(node_id):
    """Unwrap node ID if it's a single-element tuple, e.g., (5,) -> 5."""
    if isinstance(node_id, tuple) and len(node_id) == 1:
        return node_id[0]
    return node_id


class EdgeSelection(Cost):
    """Cost for edges. Handles both regular edges and hyper-edges with separate weights.

    Args:
        attribute: The attribute name (str) or tuple of attribute names to use
            for computing distance.
        regular_weight: Weight for the distance feature on regular edges.
        regular_constant: Constant cost added to each regular edge.
        hyper_weight: Weight for the distance feature on hyper-edges.
        hyper_constant: Constant cost added to each hyper-edge.
        regular_statistics: Tuple of (mean, std) for normalizing regular edge features.
        hyper_statistics: Tuple of (mean, std) for normalizing hyper-edge features.
        eps: Small value to avoid division by zero during normalization.
    """

    def __init__(
        self,
        attribute: Union[str, Tuple[str, ...]],
        regular_weight: float = 1.0,
        regular_constant: float = 0.0,
        hyper_weight: float = 0.0,
        hyper_constant: float = 0.0,
        regular_statistics: Optional[Tuple[float, float]] = None,
        hyper_statistics: Optional[Tuple[float, float]] = None,
        eps: float = 1e-8,
    ):
        self.attribute = attribute
        self.regular_weight = Weight(regular_weight)
        self.regular_constant = Weight(regular_constant)
        self.hyper_weight = Weight(hyper_weight)
        self.hyper_constant = Weight(hyper_constant)
        self.regular_statistics = regular_statistics
        self.hyper_statistics = hyper_statistics
        self.eps = eps

    def apply(self, solver: Solver) -> None:
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            is_hyper = is_hyper_edge(key)
            if is_hyper:
                feature = self._get_hyper_edge_distance(solver.graph, key)
                if self.hyper_statistics is not None:
                    mean, std = self.hyper_statistics
                    feature = (feature - mean) / (std + self.eps)
                solver.add_variable_cost(index, feature, self.hyper_weight)
                solver.add_variable_cost(index, 1.0, self.hyper_constant)
            else:
                feature = self._get_regular_edge_distance(solver.graph, key)
                if self.regular_statistics is not None:
                    mean, std = self.regular_statistics
                    feature = (feature - mean) / (std + self.eps)
                solver.add_variable_cost(index, feature, self.regular_weight)
                solver.add_variable_cost(index, 1.0, self.regular_constant)

    def _get_node_position(self, graph, node: int) -> np.ndarray:
        """Get position vector for a node based on attribute(s)."""
        if isinstance(self.attribute, tuple):
            return np.array([graph.nodes[node][attr] for attr in self.attribute])
        return np.array(graph.nodes[node][self.attribute])

    def _get_regular_edge_distance(self, graph, edge):
        """Get distance for regular edge."""
        if self.attribute in graph.edges[edge]:
            return graph.edges[edge][self.attribute]

        u, v = edge
        u = _unwrap_node_id(u)
        v = _unwrap_node_id(v)
        try:
            pos_u = self._get_node_position(graph, u)
            pos_v = self._get_node_position(graph, v)
            return np.linalg.norm(pos_u - pos_v)
        except KeyError:
            logger.warning(
                f"Attribute '{self.attribute}' not found on edge or node for edge {edge}"
            )
            return 0.0

    def _get_hyper_edge_distance(self, graph, edge):
        """Get distance for hyper-edge: ||p_source - (p_target1 + p_target2) / 2||_2."""
        if self.attribute in graph.edges[edge]:
            return graph.edges[edge][self.attribute]

        src, (tgt1, tgt2) = edge
        src = _unwrap_node_id(src)
        try:
            pos_src = self._get_node_position(graph, src)
            pos_tgt1 = self._get_node_position(graph, tgt1)
            pos_tgt2 = self._get_node_position(graph, tgt2)
            midpoint = (pos_tgt1 + pos_tgt2) / 2
            return np.linalg.norm(pos_src - midpoint)
        except KeyError:
            logger.warning(
                f"Attribute '{self.attribute}' not found on edge or node for hyper-edge {edge}"
            )
            return 0.0


class NodeSelection(Cost):
    """Cost for selecting nodes based on a node attribute.

    Nodes with low attribute values are encouraged to be selected.
    The attribute is z-score normalized if statistics are provided.

    Args:
        attribute: The attribute name to use for the node cost.
        weight: Weight for the attribute feature.
        constant: Constant cost added to each node.
        statistics: Tuple of (mean, std) for z-score normalization.
        eps: Small value to avoid division by zero during normalization.
    """

    def __init__(
        self,
        attribute: str,
        weight: float = 1.0,
        constant: float = 0.0,
        statistics: Optional[Tuple[float, float]] = None,
        eps: float = 1e-8,
    ):
        self.attribute = attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.statistics = statistics
        self.eps = eps

    def apply(self, solver: Solver) -> None:
        node_variables = solver.get_variables(NodeSelected)
        for node, index in node_variables.items():
            # Skip hypernodes (tuples used for hyper-edges)
            if isinstance(node, tuple):
                continue

            feature = self._get_node_feature(solver.graph, node)
            if self.statistics is not None:
                mean, std = self.statistics
                feature = (feature - mean) / (std + self.eps)
            solver.add_variable_cost(index, feature, self.weight)
            solver.add_variable_cost(index, 1.0, self.constant)

    def _get_node_feature(self, graph, node) -> float:
        """Get the feature value for a node."""
        try:
            value = graph.nodes[node].get(self.attribute)
            if value is None:
                logger.warning(f"Attribute '{self.attribute}' not found on node {node}")
                return 0.0
            # Handle array-like attributes by computing magnitude
            if hasattr(value, "__len__") and not isinstance(value, str):
                return float(np.linalg.norm(value))
            return float(value)
        except KeyError:
            logger.warning(f"Node {node} not found in graph")
            return 0.0
