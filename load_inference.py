from pathlib import Path

import numpy as np
import open3d as o3d


def load_prediction_ply(path):
    path = Path(path)

    line_set = o3d.io.read_line_set(str(path))

    nodes = np.asarray(line_set.points, dtype=np.float64,)
    edges = np.asarray(line_set.lines, dtype=np.int64)

    assert len(nodes) > 0, f"{path}: no nodes"
    assert nodes.ndim == 2 and nodes.shape[1] == 3, f"{path}: invalid node shape {nodes.shape}"
    assert edges.ndim == 2 and edges.shape[1] == 2, f"{path}: invalid edge shape {edges.shape}"

    return {
        "nodes": nodes,
        "edges": edges,
    }
