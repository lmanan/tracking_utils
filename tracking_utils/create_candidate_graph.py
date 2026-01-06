from pathlib import Path
import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from skimage.measure import regionprops_table
from typing import Optional, List, Dict
from data_utils import load_csv_data, load_csv_edge_attribute, load_csv_node_attribute
import zarr


def _add_nodes(G: nx.DiGraph, numerical_data: np.ndarray) -> None:
    """Add nodes to the graph from numerical data.

    Each row in numerical_data has columns: id, time, [z], y, x, p_id.
    The parent_id (last column) is ignored.
    """
    for row in numerical_data:
        node_id = int(row[0])
        attrs = {
            "time": row[1],
            "y": row[-3],
            "x": row[-2],
        }
        if len(row) > 5:
            attrs["z"] = row[2]
        G.add_node(node_id, **attrs)


def _group_nodes_by_frame(G: nx.DiGraph) -> Dict[int, List[int]]:
    """Group node IDs by their time frame."""
    frames = {}
    for node_id, attrs in G.nodes(data=True):
        t = int(attrs["time"])
        if t not in frames:
            frames[t] = []
        frames[t].append(node_id)
    return frames


def _add_edges(
    G: nx.DiGraph,
    frames: Dict[int, List[int]],
    num_neighbors: int,
    delta_t: int,
) -> None:
    """Add edges connecting nodes across time frames using KDTree spatial search.

    For each frame t, connects nodes to their k nearest spatial neighbors
    in frames t+1, t+2, ..., t+delta_t.
    """
    sample_node = list(G.nodes(data=True))[0][1]
    is_3d = "z" in sample_node

    sorted_times = sorted(frames.keys())
    for t in sorted_times:
        source_nodes = frames[t]

        for dt in range(1, delta_t + 1):
            target_t = t + dt
            if target_t not in frames:
                continue

            target_nodes = frames[target_t]

            if is_3d:
                target_coords = np.array(
                    [
                        [G.nodes[n]["z"], G.nodes[n]["y"], G.nodes[n]["x"]]
                        for n in target_nodes
                    ]
                )
            else:
                target_coords = np.array(
                    [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in target_nodes]
                )

            tree = KDTree(target_coords)

            for src in source_nodes:
                if is_3d:
                    src_coord = [
                        G.nodes[src]["z"],
                        G.nodes[src]["y"],
                        G.nodes[src]["x"],
                    ]
                else:
                    src_coord = [G.nodes[src]["y"], G.nodes[src]["x"]]

                k = min(num_neighbors, len(target_nodes))
                distances, indices = tree.query(src_coord, k=k)

                if k == 1:
                    distances = [distances]
                    indices = [indices]

                for dist, idx in zip(distances, indices):
                    tgt = target_nodes[idx]
                    G.add_edge(src, tgt, distance=dist)


def _add_region_props(
    G: nx.DiGraph,
    frames: Dict[int, List[int]],
    region_props_attributes: List[str],
    zarr_container: Path,
    group: str,
    label_dataset: str,
    raw_dataset: str,
) -> None:
    """Add region properties as node attributes from zarr label/intensity data."""
    zarr_root = zarr.open(zarr_container, mode="r")[group]
    label_data = zarr_root[label_dataset]
    intensity_data = zarr_root[raw_dataset]

    for t, node_list in frames.items():
        t_int = int(t)
        label_frame = np.array(label_data[t_int])
        intensity_frame = np.array(intensity_data[t_int])

        props = regionprops_table(
            label_frame,
            intensity_image=intensity_frame,
            properties=["label"] + region_props_attributes,
        )

        label_to_props = {}
        for i, label in enumerate(props["label"]):
            label_to_props[label] = {
                attr: props[attr][i] for attr in region_props_attributes
            }

        for node_id in node_list:
            if node_id in label_to_props:
                for attr, value in label_to_props[node_id].items():
                    G.nodes[node_id][attr] = value


def _add_edge_attributes_from_csv(
    G: nx.DiGraph, edge_attributes_csv_path: List[Path]
) -> None:
    """Add edge attributes from CSV files."""
    for edge_attr_csv in edge_attributes_csv_path:
        edge_attr_data, *_ = load_csv_edge_attribute(edge_attr_csv)
        for (src, tgt), attrs in edge_attr_data.items():
            if G.has_edge(src, tgt):
                for attr_name, attr_value in attrs.items():
                    G.edges[src, tgt][attr_name] = attr_value


def _add_node_attributes_from_csv(
    G: nx.DiGraph, node_attributes_csv_path: List[Path]
) -> None:
    """Add node attributes from CSV files."""
    for node_attr_csv in node_attributes_csv_path:
        node_attr_data, *_ = load_csv_node_attribute(node_attr_csv)
        for node_id, attrs in node_attr_data.items():
            if G.has_node(node_id):
                for attr_name, attr_value in attrs.items():
                    G.nodes[node_id][attr_name] = attr_value


def create_candidate_graph(
    num_neighbors: int,
    csv_path: Path,
    group: Optional[str] = None,
    edge_attributes_csv_path: Optional[List[Path]] = None,
    node_attributes_csv_path: Optional[List[Path]] = None,
    region_props_attributes: Optional[List[str]] = None,
    zarr_container: Optional[Path] = None,
    label_dataset: Optional[str] = None,
    raw_dataset: Optional[str] = None,
    delta_t: int = 1,
    voxel_size: Dict[str, float] = {"y": 1.0, "x": 1.0},
) -> nx.DiGraph:
    """Create a candidate graph for tracking.

    Constructs a directed graph where nodes represent cell detections and edges
    represent potential associations between cells across time frames.

    Args:
        num_neighbors: Number of nearest spatial neighbors to connect per node.
        csv_path: Path to CSV file with columns: sequence, id, time, [z], y, x, p_id.
        edge_attributes_csv_path: Optional list of CSV paths for edge attributes.
        node_attributes_csv_path: Optional list of CSV paths for node attributes
            (e.g., confidence scores, pin attributes, ground truth labels).
        region_props_attributes: Optional list of regionprops attributes to compute
            from the label image (e.g., ["area", "intensity_mean"]).
        zarr_container: Path to zarr container (required if region_props_attributes
            is specified).
        group: Group name used to filter rows in the CSV file and to access
            the corresponding group within the zarr container.
        label_dataset: Name of label dataset in zarr (required if region_props_attributes
            is specified).
        raw_dataset: Name of raw/intensity dataset in zarr (required if
            region_props_attributes is specified).
        delta_t: Number of future frames to connect. If delta_t=1, nodes at frame t
            connect to neighbors at t+1. If delta_t=2, nodes connect to neighbors
            at both t+1 and t+2. Default is 1.
        voxel_size: Dictionary with spatial scale factors for coordinates.
            Default is {"y": 1.0, "x": 1.0}.

    Returns:
        A NetworkX DiGraph with:
            - Nodes: Cell detections with attributes (time, y, x, [z], and any
              region props or CSV-provided attributes).
            - Edges: Candidate associations between cells with distance attribute
              and any CSV-provided attributes.
    """
    numerical_data, *_ = load_csv_data(
        csv_path, voxel_size=voxel_size, sequences=[group]
    )

    G = nx.DiGraph()

    _add_nodes(G, numerical_data)
    frames = _group_nodes_by_frame(G)
    _add_edges(G, frames, num_neighbors, delta_t)

    if region_props_attributes is not None:
        _add_region_props(
            G,
            frames,
            region_props_attributes,
            zarr_container,
            group,
            label_dataset,
            raw_dataset,
        )

    if edge_attributes_csv_path is not None:
        _add_edge_attributes_from_csv(G, edge_attributes_csv_path)

    if node_attributes_csv_path is not None:
        _add_node_attributes_from_csv(G, node_attributes_csv_path)

    return G
