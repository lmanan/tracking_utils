import logging
import numpy as np
from motile.costs import Cost, Weight
from motile.solver import Solver
from typing import Optional, Tuple
from motile.variables import EdgeSelected

logger = logging.getLogger(__name__)


class EdgeDistance(Cost):
    def __init__(
        self,
        attribute: str,
        weight: float,
        constant: float,
        statistics: Optional[Tuple[float, float]] = None,
    ):
        self.attribute = attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.statistics = statistics

    def apply(self, solver: Solver) -> None:
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            feature = self._get_edge_distance(solver.graph, key)
            if self.statistics is not None:
                mean, std = self.statistics
                feature = (feature - mean) / std
            solver.add_variable_cost(index, feature, self.weight)
            solver.add_variable_cost(index, 1.0, self.constant)

    def _get_edge_distance(self, graph, edge):
        if self.attribute in graph.edges[edge]:
            return graph.edges[edge][self.attribute]
        u, v = edge
        if self.attribute in graph.nodes[u]:
            u_attr = graph.nodes[u][self.attribute]
            v_attr = graph.nodes[v][self.attribute]
            return np.linalg.norm(np.array(u_attr) - np.array(v_attr))
        logger.warning(
            f"Attribute '{self.attribute}' not found on edge or node for edge {edge}"
        )
        return 0.0
