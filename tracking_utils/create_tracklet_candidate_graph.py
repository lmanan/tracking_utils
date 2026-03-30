import logging
import re
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from data_utils.load_tracklet_csv_data import load_tracklet_csv_data
from data_utils.load_tracklet_csv_node_attribute import load_tracklet_csv_node_attribute

logger = logging.getLogger(__name__)


def _load_tracklet_ids(file_path: Path, delimiter: str = " ") -> np.ndarray:
    """Read unique tracklet_ids from a CSV file."""
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(delimiter)
    col = header.index("tracklet_id")
    data = np.genfromtxt(file_path, delimiter=delimiter, skip_header=1, usecols=[col])
    return np.unique(data.astype(np.int64))


def _build_embeddings_lookup(
    embeddings_data: np.ndarray, valid_ids: np.ndarray
) -> dict:
    """Build a dict mapping tracklet_id -> (k, D) embedding matrix.

    Args:
        embeddings_data: Structured array with fields (tracklet_id, time, k, emb_0, ...).
        valid_ids: Array of valid tracklet IDs to include.

    Returns:
        dict mapping tracklet_id (int) -> np.ndarray of shape (k, D).
    """
    emb_fields = [
        f
        for f in embeddings_data.dtype.names
        if f not in ("tracklet_id", "time", "t", "k")
    ]
    lookup = {}
    for tid in valid_ids:
        mask = embeddings_data["tracklet_id"] == tid
        rows = embeddings_data[mask]
        lookup[int(tid)] = np.stack([rows[f] for f in emb_fields], axis=1)  # (k, D)
    return lookup


def _get_tracklet_endpoints(
    tracklet_data: np.ndarray,
    valid_ids: np.ndarray,
    t_min: Optional[int] = None,
    t_max: Optional[int] = None,
) -> dict:
    """For each valid tracklet_id, compute first and last detection time, centroid, and per-keypoint positions.

    If t_min/t_max are provided, each tracklet is clipped to [t_min, t_max]: only
    detections within the window are considered, so t_start/t_end and the first/last
    keypoint positions reflect the clipped range. Tracklets with no detections inside
    the window are excluded.

    Returns:
        dict mapping tracklet_id -> {
            t_start, first_y, first_x, first_kps,
            t_end,   last_y,  last_x,  last_kps,
        }
        where first_kps/last_kps are (n_keypoints, 2) arrays of [y, x] per keypoint.
    """
    field_names = tracklet_data.dtype.names
    kp_pattern = re.compile(r"^kp(\d+)_(y|x)$")
    kp_nums = sorted(
        {int(kp_pattern.match(n).group(1)) for n in field_names if kp_pattern.match(n)}
    )
    kp_y_cols = [f"kp{i}_y" for i in kp_nums]
    kp_x_cols = [f"kp{i}_x" for i in kp_nums]
    time_col = next(name for name in field_names if name in ("time", "t"))

    centroid_y = np.stack([tracklet_data[c] for c in kp_y_cols], axis=1).mean(axis=1)
    centroid_x = np.stack([tracklet_data[c] for c in kp_x_cols], axis=1).mean(axis=1)

    endpoints = {}
    for tid in valid_ids:
        mask = tracklet_data["tracklet_id"] == tid
        times = tracklet_data[time_col][mask]

        # Clip to [t_min, t_max]: only consider detections within the window
        window_mask = np.ones(len(times), dtype=bool)
        if t_min is not None:
            window_mask &= times >= t_min
        if t_max is not None:
            window_mask &= times <= t_max
        if not window_mask.any():
            continue  # no detections in window, skip this tracklet

        times = times[window_mask]
        ys = centroid_y[mask][window_mask]
        xs = centroid_x[mask][window_mask]
        rows_in_window = tracklet_data[mask][window_mask]

        first_idx = np.argmin(times)
        last_idx = np.argmax(times)

        def _kps(row_idx, rows=rows_in_window):
            ys = np.array([rows[c][row_idx] for c in kp_y_cols])
            xs = np.array([rows[c][row_idx] for c in kp_x_cols])
            return np.stack([ys, xs], axis=1)  # (n_kp, 2)

        endpoints[int(tid)] = {
            "t_start": int(times[first_idx]),
            "first_y": float(ys[first_idx]),
            "first_x": float(xs[first_idx]),
            "first_kps": _kps(first_idx),
            "t_end": int(times[last_idx]),
            "last_y": float(ys[last_idx]),
            "last_x": float(xs[last_idx]),
            "last_kps": _kps(last_idx),
        }

    return endpoints, kp_nums


def create_tracklet_candidate_graph(
    tracklet_csv_path: Path,
    tracklet_embeddings_path: Path,
    num_neighbors: int,
    max_spatial_distance: Optional[float],
    max_time_gap: int,
    embeddings_attribute_prefix: str = "emb",
    t_min: Optional[int] = None,
    t_max: Optional[int] = None,
    max_time_gap_past: int = 0,
) -> nx.DiGraph:
    """Create a candidate graph for tracklet stitching.

    Nodes are tracklets present in both the tracklet CSV and the embeddings CSV.
    Candidate edges connect the last detection of a source tracklet to the first
    detection of a target tracklet, subject to:
        - The target's first detection time falls in the window
          (source.t_end - max_time_gap_past, source.t_end + max_time_gap).
          With max_time_gap_past=0 (default) only forward edges are allowed.
        - The edge direction in motile's frame ordering is preserved:
          target.t_start > source.t_start.
        - The spatial distance between source's last centroid and target's
          first centroid is <= max_spatial_distance (if set).
        - At most num_neighbors edges are added per source (closest centroid first).

    Each edge carries the following attributes:
        - kp_{i}_distance: Euclidean distance between the i-th keypoint of the
          source's last detection and the target's first detection.
        - id_distance: Mean L2 distance over the (k_src x k_tgt) matrix of
          pairwise embedding distances between the two tracklets.
        - t_gap: |target.t_start - source.t_end| (always non-negative).

    Args:
        tracklet_csv_path: Path to the tracklet CSV file
            (columns: tracklet_id, time, kp0_y, kp0_x, ...).
        tracklet_embeddings_path: Path to the tracklet embeddings CSV file
            (columns: tracklet_id, time, k, <prefix>_0, ...).
        num_neighbors: Maximum number of candidate edges per source tracklet.
        max_spatial_distance: Maximum centroid distance for candidate edges.
            If None, no spatial limit is applied.
        max_time_gap: Maximum allowed forward time gap (exclusive) between the
            last detection of a source and the first detection of a target.
        embeddings_attribute_prefix: Prefix used to identify embedding columns
            in the embeddings CSV. Defaults to 'emb'.
        t_min: Optional minimum time (inclusive). If provided, only tracklets
            with t_start >= t_min are included. Default is None (no lower bound).
        t_max: Optional maximum time (inclusive). If provided, only tracklets
            with t_end <= t_max are included. Default is None (no upper bound).
        max_time_gap_past: Maximum allowed backward overlap (exclusive) between
            the last detection of a source and the first detection of a target.
            A value of 0 (default) disallows backward edges. A value of N allows
            the target to start up to N frames before the source ends.

    Returns:
        nx.DiGraph where each node is a tracklet_id with attributes
        (t_start, first_y, first_x, t_end, last_y, last_x) and each
        edge has attributes (kp_{i}_distance, id_distance, t_gap).
    """
    tracklet_data = load_tracklet_csv_data(tracklet_csv_path)
    embeddings_data = load_tracklet_csv_node_attribute(
        tracklet_embeddings_path, attribute_prefix=embeddings_attribute_prefix
    )

    ids_from_tracklets = np.unique(tracklet_data["tracklet_id"])
    ids_from_embeddings = np.unique(embeddings_data["tracklet_id"])
    valid_ids = np.intersect1d(ids_from_tracklets, ids_from_embeddings)
    logger.info("Found %d valid tracklet IDs.", len(valid_ids))

    endpoints, kp_nums = _get_tracklet_endpoints(
        tracklet_data, valid_ids, t_min=t_min, t_max=t_max
    )
    valid_ids = np.array(list(endpoints.keys()), dtype=np.int64)
    if t_min is not None and t_max is not None:
        tracklet_summary = "\n".join(
            f"  {tid} (t={endpoints[tid]['t_start']}-{endpoints[tid]['t_end']},"
            f" start_xy=({endpoints[tid]['first_x']:.1f},{endpoints[tid]['first_y']:.1f}),"
            f" end_xy=({endpoints[tid]['last_x']:.1f},{endpoints[tid]['last_y']:.1f}))"
            for tid in sorted(valid_ids.tolist())
        )
        logger.info(
            "After t_min/t_max clipping: %d tracklet IDs:\n%s",
            len(valid_ids),
            tracklet_summary,
        )
    else:
        logger.info("After t_min/t_max clipping: %d tracklet IDs.", len(valid_ids))

    embeddings_lookup = _build_embeddings_lookup(embeddings_data, valid_ids)

    # Add nodes (exclude internal-only keys first_kps/last_kps)
    G = nx.DiGraph()
    for tid, ep in endpoints.items():
        G.add_node(
            tid,
            t_start=ep["t_start"],
            first_y=ep["first_y"],
            first_x=ep["first_x"],
            t_end=ep["t_end"],
            last_y=ep["last_y"],
            last_x=ep["last_x"],
            tracklet_length=ep["t_end"] - ep["t_start"] + 1,
        )

    # Arrays over all targets for vectorised filtering
    all_ids = np.array(list(endpoints.keys()))
    t_starts = np.array([endpoints[tid]["t_start"] for tid in all_ids])
    first_coords = np.array(
        [[endpoints[tid]["first_y"], endpoints[tid]["first_x"]] for tid in all_ids]
    )

    for src_id, src_ep in endpoints.items():
        src_t_end = src_ep["t_end"]
        src_last_pos = np.array([src_ep["last_y"], src_ep["last_x"]])
        src_last_kps = src_ep["last_kps"]  # (n_kp, 2)
        src_embs = embeddings_lookup[src_id]  # (k, D)

        # Filter targets: within (-max_time_gap_past, max_time_gap) of source.t_end,
        # and target.t_start > source.t_start (preserve motile's forward-time ordering).
        src_t_start = src_ep["t_start"]
        t_gaps = t_starts - src_t_end
        time_mask = (t_gaps > -max_time_gap_past) & (t_gaps < max_time_gap)
        time_mask &= t_starts > src_t_start  # motile requires edges go forward in frame
        time_mask &= all_ids != src_id

        candidate_indices = np.where(time_mask)[0]
        if len(candidate_indices) == 0:
            continue

        candidate_coords = first_coords[candidate_indices]
        k = min(num_neighbors, len(candidate_indices))
        tree = KDTree(candidate_coords)
        distances, indices = tree.query(src_last_pos, k=k)

        if k == 1:
            distances = [distances]
            indices = [indices]

        for dist, idx in zip(distances, indices):
            if max_spatial_distance is not None and dist > max_spatial_distance:
                continue

            tgt_id = int(all_ids[candidate_indices[idx]])
            tgt_ep = endpoints[tgt_id]
            tgt_first_kps = tgt_ep["first_kps"]  # (n_kp, 2)
            tgt_embs = embeddings_lookup[tgt_id]  # (k, D)

            # Per-keypoint distances
            kp_distances = {
                f"kp_{i}_distance": float(
                    np.linalg.norm(src_last_kps[out_idx] - tgt_first_kps[out_idx])
                )
                for out_idx, i in enumerate(kp_nums)
            }

            # id_distance: mean over (k_src x k_tgt) pairwise L2 distance matrix
            diff = src_embs[:, None, :] - tgt_embs[None, :, :]  # (k_s, k_t, D)
            id_distance = float(np.linalg.norm(diff, axis=2).mean())

            t_gap = abs(tgt_ep["t_start"] - src_t_end)

            G.add_edge(
                src_id,
                tgt_id,
                **kp_distances,
                id_distance=id_distance,
                t_gap=t_gap,
            )

    logger.info(
        "Candidate graph: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges()
    )

    # Diagnostic: active tracklets per frame
    t_starts = np.array([ep["t_start"] for ep in endpoints.values()])
    t_ends = np.array([ep["t_end"] for ep in endpoints.values()])
    all_frames = np.arange(t_starts.min(), t_ends.max() + 1)
    active_per_frame = np.array(
        [np.sum((t_starts <= t) & (t_ends >= t)) for t in all_frames]
    )
    logger.info(
        "Active tracklets per frame — min: %d, max: %d, mean: %.1f",
        active_per_frame.min(),
        active_per_frame.max(),
        active_per_frame.mean(),
    )

    return G
