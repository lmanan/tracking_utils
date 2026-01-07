import networkx as nx
import numpy as np
from typing import Dict, List, Tuple


def _compute_mean_std(values: List[float]) -> Tuple[float, float]:
    if values:
        mean_val, std_val = np.mean(values), np.std(values)
        return mean_val, std_val
    return None, None


def compute_graph_statistics(
    graph: nx.DiGraph, attributes: List[str]
) -> Dict[str, Tuple[float, float]]:
    statistics = {}

    # Sample an edge and node to check where attributes live
    sample_edge_data = next(iter(graph.edges(data=True)), (None, None, {}))[2]
    sample_node_data = next(iter(graph.nodes(data=True)), (None, {}))[1]

    for attr in attributes:
        if attr in sample_edge_data:
            # Edge attribute - directly take values
            values = [data.get(attr, 0.0) for _, _, data in graph.edges(data=True)]
        elif attr in sample_node_data:
            # Node attribute - compute norm distance on edge endpoints
            values = []
            for u, v in graph.edges():
                u_attr = graph.nodes[u].get(attr, 0.0)
                v_attr = graph.nodes[v].get(attr, 0.0)
                values.append(np.linalg.norm(np.array(u_attr) - np.array(v_attr)))
        else:
            statistics[attr] = (None, None)
            continue

        mean_val, std_val = _compute_mean_std(values)
        statistics[attr] = (mean_val, std_val)

    return statistics
