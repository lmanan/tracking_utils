import networkx as nx
import numpy as np
from typing import Dict, List, Tuple


def _compute_mean_std(values: List[float]) -> Tuple[float, float]:
    if values:
        mean_val, std_val = np.mean(values), np.std(values)
        return mean_val, std_val
    return None, None


def _is_hypernode(graph: nx.DiGraph, node, frame_attribute: str = "time") -> bool:
    """Check if a node is a hypernode (no frame attribute, used for hyper-edges)."""
    return frame_attribute not in graph.nodes.get(node, {})


def compute_graph_statistics(
    graph: nx.DiGraph,
    attributes: List[str],
    frame_attribute: str = "time",
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Compute mean and std for edge attributes, separately for regular and hyper edges.

    Args:
        graph: The candidate graph with regular and/or hyper edges.
            Hyper-edges are represented as hypernodes: src -> hypernode -> tgt1, tgt2
        attributes: List of attribute names to compute statistics for.
        frame_attribute: The attribute name used for frame/time. Nodes without this
            attribute are considered hypernodes.

    Returns:
        Dictionary mapping attribute names to a dict with 'regular' and 'hyper' keys,
        each containing (mean, std) tuples.
    """
    statistics = {}

    # Sample a real node (not hypernode) to check where attributes live
    sample_node_data = {}
    for node, data in graph.nodes(data=True):
        if frame_attribute in data:
            sample_node_data = data
            break

    # Separate regular edges and hyper-edges (edges going into hypernodes)
    regular_edges = []
    hyper_edges = []  # List of (src, hypernode, [tgt1, tgt2, ...], edge_data)

    for u, v, data in graph.edges(data=True):
        if _is_hypernode(graph, v, frame_attribute):
            # Edge from src to hypernode - this is a hyper-edge
            hypernode = v
            targets = list(graph.successors(hypernode))
            hyper_edges.append((u, hypernode, targets, data))
        elif not _is_hypernode(graph, u, frame_attribute):
            # Regular edge (neither endpoint is a hypernode)
            regular_edges.append((u, v, data))
        # Skip edges from hypernode to targets (already captured above)

    # Sample edge data from each type
    sample_regular_edge_data = regular_edges[0][2] if regular_edges else {}
    sample_hyper_edge_data = hyper_edges[0][3] if hyper_edges else {}

    for attr in attributes:
        statistics[attr] = {"regular": (None, None), "hyper": (None, None)}

        # Check if attribute is on edges or nodes
        is_edge_attr = (
            attr in sample_regular_edge_data or attr in sample_hyper_edge_data
        )
        is_node_attr = attr in sample_node_data

        if not is_edge_attr and not is_node_attr:
            continue

        # Compute statistics for regular edges
        if regular_edges:
            if is_edge_attr:
                regular_values = [data.get(attr, 0.0) for _, _, data in regular_edges]
            else:
                regular_values = []
                for u, v, _ in regular_edges:
                    u_attr = graph.nodes[u].get(attr, 0.0)
                    v_attr = graph.nodes[v].get(attr, 0.0)
                    regular_values.append(
                        np.linalg.norm(np.array(u_attr) - np.array(v_attr))
                    )
            statistics[attr]["regular"] = _compute_mean_std(regular_values)

        # Compute statistics for hyper edges
        if hyper_edges:
            if is_edge_attr:
                hyper_values = [data.get(attr, 0.0) for _, _, _, data in hyper_edges]
            else:
                # For node attributes on hyper edges, compute distance from source
                # to midpoint of targets
                hyper_values = []
                for src, hypernode, targets, _ in hyper_edges:
                    if len(targets) >= 2:
                        src_attr = np.array(graph.nodes[src].get(attr, 0.0))
                        tgt_attrs = [
                            np.array(graph.nodes[t].get(attr, 0.0)) for t in targets
                        ]
                        midpoint = np.mean(tgt_attrs, axis=0)
                        hyper_values.append(np.linalg.norm(src_attr - midpoint))
            statistics[attr]["hyper"] = _compute_mean_std(hyper_values)

    return statistics
