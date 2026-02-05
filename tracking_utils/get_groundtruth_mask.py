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
    mode: Literal["sparse", "dense"] = "dense",
):
    """Build groundtruth and mask arrays over solver variables.

    Args:
        solver: The motile Solver instance with a solved or unsolved graph.
        gt_attribute: Name of the attribute on nodes/edges that holds the
            ground truth value.
        mode: "sparse" or "dense".
            - sparse: Only annotated edges/nodes (where gt_attribute is present)
              are included in the mask. Assumes that if a node is splitting,
              both daughter edges are annotated.
            - dense: Every edge and node is expected to have the gt_attribute.

    Returns:
        A tuple (groundtruth, mask) of float32 arrays of length
        solver.num_variables.
    """
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
    else:  # dense
        splitting_nodes = set()
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
                    outgoing_nodes.add(u)
                    incoming_nodes.add(v)

        for node, index in solver.get_variables(NodeSelected).items():
            gt = solver.graph.nodes[node][gt_attribute]
            mask[index] = 1.0
            groundtruth[index] = gt

        for node, index in solver.get_variables(NodeSplit).items():
            if node in splitting_nodes:
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
