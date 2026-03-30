import logging

import networkx as nx
import numpy as np
import motile
from motile.variables import EdgeSelected

logger = logging.getLogger(__name__)


def log_average_occupancy(solution_graph: nx.DiGraph, num_tracks: int) -> None:
    """Log two occupancy metrics:

    - Tracklet occupancy: fraction of frames covered by an active tracklet,
      averaged over all frames and divided by num_tracks. This is < 1 when
      there are intra-chain gaps between stitched tracklets.

    - Chain occupancy: fraction of frames within the span [chain_t_start,
      chain_t_end] of each stitched chain, averaged over all frames and
      divided by num_tracks. Treats the entire chain span as covered,
      ignoring intra-chain gaps. This reaches 1 when all chains span the
      full sequence.

    Args:
        solution_graph: Solution graph where each node has t_start and t_end attributes.
        num_tracks: Expected number of active tracks per frame (used as denominator).
    """
    if solution_graph.number_of_nodes() == 0:
        logger.warning("Solution graph is empty, cannot compute occupancy.")
        return

    # --- Tracklet occupancy ---
    t_starts = np.array(
        [solution_graph.nodes[n]["t_start"] for n in solution_graph.nodes()]
    )
    t_ends = np.array(
        [solution_graph.nodes[n]["t_end"] for n in solution_graph.nodes()]
    )
    all_frames = np.arange(t_starts.min(), t_ends.max() + 1)
    tracklet_occupancy_per_frame = (
        np.array([np.sum((t_starts <= t) & (t_ends >= t)) for t in all_frames])
        / num_tracks
    )
    logger.info("Average tracklet occupancy: %.4f", tracklet_occupancy_per_frame.mean())

    # --- Chain occupancy ---
    # Walk each chain and record its full span [chain_t_start, chain_t_end]
    parent_lookup = {dst: src for src, dst in solution_graph.edges()}
    roots = [n for n in solution_graph.nodes() if n not in parent_lookup]
    chain_t_starts, chain_t_ends = [], []
    for root in roots:
        node = root
        chain_nodes = []
        while node is not None:
            chain_nodes.append(node)
            children = list(solution_graph.successors(node))
            node = children[0] if children else None
        chain_t_starts.append(solution_graph.nodes[chain_nodes[0]]["t_start"])
        chain_t_ends.append(solution_graph.nodes[chain_nodes[-1]]["t_end"])

    chain_t_starts = np.array(chain_t_starts)
    chain_t_ends = np.array(chain_t_ends)
    chain_occupancy_per_frame = (
        np.array(
            [np.sum((chain_t_starts <= t) & (chain_t_ends >= t)) for t in all_frames]
        )
        / num_tracks
    )
    logger.info("Average chain occupancy: %.4f", chain_occupancy_per_frame.mean())


def log_solution_tracks(solution_graph: nx.DiGraph) -> None:
    """Log the start and end frame for each track in the solution.

    A track is a chain of stitched tracklets connected by edges in the
    solution graph. For each chain, logs the t_start of the first tracklet
    and the t_end of the last tracklet.

    Args:
        solution_graph: Solution graph where each node has t_start and t_end attributes.
    """
    if solution_graph.number_of_nodes() == 0:
        logger.warning("Solution graph is empty, cannot log solution tracks.")
        return
    parent_lookup = {dst: src for src, dst in solution_graph.edges()}
    roots = [n for n in solution_graph.nodes() if n not in parent_lookup]
    for chain_id, root in enumerate(sorted(roots), start=1):
        chain = []
        node = root
        while node is not None:
            chain.append(node)
            children = list(solution_graph.successors(node))
            node = children[0] if children else None
        chain_t_start = solution_graph.nodes[chain[0]]["t_start"]
        chain_t_end = solution_graph.nodes[chain[-1]]["t_end"]
        logger.info(
            "Track %d: t_start=%d, t_end=%d", chain_id, chain_t_start, chain_t_end
        )


def log_edge_margin_ranking(
    solver: motile.Solver,
    solution_graph: nx.DiGraph,
    candidate_graph: nx.DiGraph,
) -> None:
    """For each solution node with a selected outgoing edge, compute the margin
    between the selected edge cost and the next-best outgoing edge cost.
    Logs nodes sorted ascending by margin (lowest margin = most uncertain = proofread first).

    Args:
        solver: The solved motile Solver instance.
        solution_graph: The selected solution subgraph.
        candidate_graph: The full candidate graph.
    """
    weights = solver.weights.to_ndarray()
    features = solver.features.to_ndarray()
    all_costs = features @ weights  # cost per variable

    edge_indicators = solver.get_variables(EdgeSelected)  # edge -> variable index

    rows = []
    for u in solution_graph.nodes():
        # Only consider nodes with a selected outgoing edge
        selected_out = list(solution_graph.successors(u))
        if not selected_out:
            continue
        selected_edge = (u, selected_out[0])
        selected_cost = all_costs[edge_indicators[selected_edge]]

        # All outgoing edges from u in the candidate graph
        all_out_edges = [(u, v) for v in candidate_graph.successors(u)]
        if len(all_out_edges) < 2:
            continue  # no runner-up to compare against

        other_costs = [
            all_costs[edge_indicators[e]]
            for e in all_out_edges
            if e != selected_edge and e in edge_indicators
        ]
        if not other_costs:
            continue

        runner_up_cost = min(other_costs)  # next-best = lowest cost among non-selected
        margin = runner_up_cost - selected_cost

        node_attrs = solution_graph.nodes[u]
        rows.append(
            {
                "tracklet_id": u,
                "t_start": node_attrs.get("t_start"),
                "t_end": node_attrs.get("t_end"),
                "last_y": node_attrs.get("last_y"),
                "last_x": node_attrs.get("last_x"),
                "margin": margin,
            }
        )

    rows.sort(key=lambda r: r["margin"])

    logger.info(
        "Edge margin ranking (%d nodes with outgoing edges), sorted by margin ascending:",
        len(rows),
    )
    logger.info(
        "%-15s %-8s %-8s %-10s %-10s %-10s",
        "tracklet_id",
        "t_start",
        "t_end",
        "last_y",
        "last_x",
        "margin",
    )
    for r in rows:
        logger.info(
            "%-15s %-8s %-8s %-10.1f %-10.1f %-10.4f",
            r["tracklet_id"],
            r["t_start"],
            r["t_end"],
            r["last_y"] or 0.0,
            r["last_x"] or 0.0,
            r["margin"],
        )
