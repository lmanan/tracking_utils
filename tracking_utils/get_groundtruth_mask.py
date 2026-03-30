from typing import Literal

import motile
import numpy as np
from motile.variables import (
    EdgeSelected,
    NodeAppear,
    NodeDisappear,
    NodeSelected,
    NodeSplit,
)


def get_groundtruth_mask(
    solver: motile.Solver,
    gt_attribute: str = "GT",
    mode: Literal["sparse", "full_lineage", "dense"] = "dense",
):
    """Build groundtruth and mask arrays over solver variables.

    Args:
        solver: The motile Solver instance with a solved or unsolved graph.
        gt_attribute: Name of the attribute on nodes/edges that holds the
            ground truth value.
        mode: "sparse", "full_lineage", or "dense".
            - sparse: Only annotated edges/nodes (where gt_attribute is present)
              are included in the mask. Assumes that if a node is splitting,
              both daughter edges are annotated. NodeDisappear/NodeAppear are
              only annotated as GT=0 for nodes adjacent to a GT=1 edge: the
              source node of a GT=1 edge is known not to disappear, and the
              target node is known not to appear. Lineage boundary nodes
              (source with no outgoing GT=1 edge, target with no incoming
              GT=1 edge) are left out of the mask.
            - full_lineage: Like sparse, but assumes each annotated lineage is
              fully labelled from start to end. Boundary nodes are therefore
              also annotated: a GT=1 node with no outgoing GT=1 edge gets
              NodeDisappear=1, and a GT=1 node with no incoming GT=1 edge gets
              NodeAppear=1.
            - dense: Every edge and node is expected to have the gt_attribute.

    Returns:
        A tuple (groundtruth, mask) of float32 arrays of length
        solver.num_variables.
    """
    # Pre-register all variable types so solver.num_variables is final before
    # allocating the mask.  get_variables() registers lazily on first call, so
    # calling it here prevents an IndexError when NodeSplit (or any other type)
    # is registered later and its indices exceed the pre-allocated mask size.
    for _var_type in (EdgeSelected, NodeSelected, NodeAppear, NodeDisappear, NodeSplit):
        solver.get_variables(_var_type)

    mask = np.zeros((solver.num_variables), dtype=np.float32)
    groundtruth = np.zeros_like(mask)
    if mode == "sparse":
        for edge, index in solver.get_variables(EdgeSelected).items():
            u, v = edge
            gt = solver.graph.edges[edge].get(gt_attribute, None)
            if isinstance(u, tuple):
                # hyper edge: u = (src,), v = (tgt1, tgt2)
                src = u[0]
                daughters = list(v)
                if gt is not None:
                    mask[index] = 1.0
                    groundtruth[index] = gt
                    if gt == 1.0:
                        mask[solver.get_variables(NodeDisappear)[src]] = 1.0
                        groundtruth[solver.get_variables(NodeDisappear)[src]] = 0
                        for d in daughters:
                            mask[solver.get_variables(NodeAppear)[d]] = 1.0
                            groundtruth[solver.get_variables(NodeAppear)[d]] = 0
            else:
                # regular edge
                if gt is not None:
                    mask[index] = 1.0
                    groundtruth[index] = gt
                    if gt == 1.0:
                        mask[solver.get_variables(NodeDisappear)[u]] = 1.0
                        groundtruth[solver.get_variables(NodeDisappear)[u]] = 0
                        mask[solver.get_variables(NodeAppear)[v]] = 1.0
                        groundtruth[solver.get_variables(NodeAppear)[v]] = 0

        for node, index in solver.get_variables(NodeSelected).items():
            gt = solver.graph.nodes[node].get(gt_attribute, None)
            if gt is not None:
                mask[index] = 1.0
                groundtruth[index] = gt

        splitting_nodes = set()
        outgoing_count = {}
        for edge, index in solver.get_variables(EdgeSelected).items():
            u, v = edge
            gt = solver.graph.edges[edge].get(gt_attribute, None)
            if gt == 1.0:
                if isinstance(u, tuple):
                    # a GT hyper edge means src is splitting by definition
                    splitting_nodes.add(u[0])
                else:
                    outgoing_count[u] = outgoing_count.get(u, 0) + 1

        for node, index in solver.get_variables(NodeSplit).items():
            if node in splitting_nodes or outgoing_count.get(node, 0) > 1:
                mask[index] = 1.0
                groundtruth[index] = 1.0
            else:
                mask[index] = 1.0
                groundtruth[index] = 0.0
    elif mode == "full_lineage":
        # Phase 1: identical to sparse — edges, NodeSelected, NodeSplit,
        # and NodeDisappear/NodeAppear=0 for nodes adjacent to GT=1 edges.
        gt_outgoing = set()  # sources of GT=1 edges
        gt_incoming = set()  # targets of GT=1 edges
        splitting_nodes = set()
        outgoing_count = {}

        for edge, index in solver.get_variables(EdgeSelected).items():
            u, v = edge
            gt = solver.graph.edges[edge].get(gt_attribute, None)
            if isinstance(u, tuple):
                # hyper edge: u = (src,), v = (tgt1, tgt2)
                src = u[0]
                daughters = list(v)
                if gt is not None:
                    mask[index] = 1.0
                    groundtruth[index] = gt
                    if gt == 1.0:
                        gt_outgoing.add(src)
                        for d in daughters:
                            gt_incoming.add(d)
                        splitting_nodes.add(src)
                        mask[solver.get_variables(NodeDisappear)[src]] = 1.0
                        groundtruth[solver.get_variables(NodeDisappear)[src]] = 0.0
                        for d in daughters:
                            mask[solver.get_variables(NodeAppear)[d]] = 1.0
                            groundtruth[solver.get_variables(NodeAppear)[d]] = 0.0
            else:
                # regular edge
                if gt is not None:
                    mask[index] = 1.0
                    groundtruth[index] = gt
                    if gt == 1.0:
                        gt_outgoing.add(u)
                        gt_incoming.add(v)
                        outgoing_count[u] = outgoing_count.get(u, 0) + 1
                        mask[solver.get_variables(NodeDisappear)[u]] = 1.0
                        groundtruth[solver.get_variables(NodeDisappear)[u]] = 0.0
                        mask[solver.get_variables(NodeAppear)[v]] = 1.0
                        groundtruth[solver.get_variables(NodeAppear)[v]] = 0.0

        for node, index in solver.get_variables(NodeSelected).items():
            gt = solver.graph.nodes[node].get(gt_attribute, None)
            if gt is not None:
                mask[index] = 1.0
                groundtruth[index] = gt

        for node, index in solver.get_variables(NodeSplit).items():
            if node in splitting_nodes or outgoing_count.get(node, 0) > 1:
                mask[index] = 1.0
                groundtruth[index] = 1.0
            else:
                mask[index] = 1.0
                groundtruth[index] = 0.0

        # Phase 2: annotate lineage boundaries.
        # A GT=1 node with no outgoing GT=1 edge is a lineage root → NodeDisappear=1.
        # A GT=1 node with no incoming GT=1 edge is a lineage leaf → NodeAppear=1.
        for node, index in solver.get_variables(NodeSelected).items():
            gt = solver.graph.nodes[node].get(gt_attribute, None)
            if gt == 1.0:
                if node not in gt_outgoing:
                    mask[solver.get_variables(NodeDisappear)[node]] = 1.0
                    groundtruth[solver.get_variables(NodeDisappear)[node]] = 1.0
                if node not in gt_incoming:
                    mask[solver.get_variables(NodeAppear)[node]] = 1.0
                    groundtruth[solver.get_variables(NodeAppear)[node]] = 1.0
    else:  # dense
        splitting_nodes = set()
        outgoing_count = {}
        outgoing_nodes = set()
        incoming_nodes = set()

        for edge, index in solver.get_variables(EdgeSelected).items():
            u, v = edge
            gt = solver.graph.edges[edge][gt_attribute]
            mask[index] = 1.0
            groundtruth[index] = gt
            if gt == 1.0:
                if isinstance(u, tuple):
                    # hyper edge: u = (src,), v = (tgt1, tgt2)
                    src = u[0]
                    splitting_nodes.add(src)
                    outgoing_nodes.add(src)
                    for d in v:
                        incoming_nodes.add(d)
                else:
                    outgoing_count[u] = outgoing_count.get(u, 0) + 1
                    outgoing_nodes.add(u)
                    incoming_nodes.add(v)

        for node, index in solver.get_variables(NodeSelected).items():
            gt = solver.graph.nodes[node][gt_attribute]
            mask[index] = 1.0
            groundtruth[index] = gt

        for node, index in solver.get_variables(NodeSplit).items():
            if node in splitting_nodes or outgoing_count.get(node, 0) > 1:
                mask[index] = 1.0
                groundtruth[index] = 1.0
            else:
                mask[index] = 1.0
                groundtruth[index] = 0.0
        for node, index in solver.get_variables(NodeAppear).items():
            if node not in incoming_nodes:
                mask[index] = 1.0
                groundtruth[index] = 1.0
            else:
                mask[index] = 1.0
                groundtruth[index] = 0.0
        for node, index in solver.get_variables(NodeDisappear).items():
            if node not in outgoing_nodes:
                mask[index] = 1.0
                groundtruth[index] = 1.0
            else:
                mask[index] = 1.0
                groundtruth[index] = 0.0

    return groundtruth, mask
