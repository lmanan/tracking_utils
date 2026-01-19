import networkx as nx
from motile.solver import Solver
from motile.variables import NodeSelected, EdgeSelected

from tracking_utils.utils import is_hyper_edge


def _unwrap_node_id(node_id):
    """Unwrap node ID if it's a single-element tuple, e.g., (5,) -> 5."""
    if isinstance(node_id, tuple) and len(node_id) == 1:
        return node_id[0]
    return node_id


def get_solution_graph(solver: Solver, solution) -> nx.DiGraph:
    """Extract the solution graph from a solved motile Solver.

    Handles both regular edges and hyper-edges. For hyper-edges ((src,), (tgt1, tgt2)),
    creates two regular edges: (src, tgt1) and (src, tgt2).

    Args:
        solver: The solved motile Solver instance.
        solution: The solution returned by solver.solve().

    Returns:
        A NetworkX DiGraph containing only the selected nodes and edges.
    """
    graph = solver.graph

    solution_graph = nx.DiGraph()

    # Add selected nodes
    node_indicators = solver.get_variables(NodeSelected)
    for node_id, index in node_indicators.items():
        if solution[index] > 0.5:
            unwrapped_id = _unwrap_node_id(node_id)
            if isinstance(unwrapped_id, int):
                solution_graph.add_node(unwrapped_id, **graph.nodes[node_id])

    # Add selected edges
    edge_indicators = solver.get_variables(EdgeSelected)
    for edge, index in edge_indicators.items():
        if solution[index] > 0.5:
            src, tgt = edge
            src = _unwrap_node_id(src)

            if is_hyper_edge(edge):
                # Hyper-edge: ((src,), (tgt1, tgt2)) -> create two regular edges
                tgt1, tgt2 = tgt
                edge_attrs = graph.edges[edge]
                solution_graph.add_edge(src, tgt1, **edge_attrs)
                solution_graph.add_edge(src, tgt2, **edge_attrs)
            elif isinstance(src, int) and isinstance(tgt, int):
                # Regular edge
                solution_graph.add_edge(src, tgt, **graph.edges[edge])

    return solution_graph
