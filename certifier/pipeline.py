"""
Corridor constants and helpers used by the certifier (certify3d.py, hazard.py, exact/export_cells.py).

The snake corridor of Google DeepMind's `the_3d_moving_sofa.ipynb` (AlphaEvolve repository of problems):
width W = 1, middle segment length MID = 4, initial block [-L0, 0] x [0, W]^2 with L0 = 4, fixed start pose
(-3, 0, 0, 0, 0, 0) and end pose (0, 6, 8, 90, 90, 90).  `inside` is exactly the notebook's are_points_in_corridor
(inclusive bounds); `add_start_and_end_poses` and `smooth_path` follow the notebook's functions of the same purpose.
(Only the parts of the original working file that the certifier imports are included here.)
"""
import numpy as np
from numba import njit

W = 1.0; MID = 4.0; L0 = 4.0; START_END = 3.0
YT = W + MID

def add_start_and_end_poses(path):
    return np.vstack([[-START_END, 0, 0, 0, 0, 0], path, [0.0, 2 * W + MID, W + START_END + L0, 90.0, 90.0, 90.0]])

def smooth_path(path, ttol, rtol):
    out = [path[0]]
    for p1, p2 in zip(path[:-1], path[1:]):
        td = np.linalg.norm(p2[:3] - p1[:3]); rd = np.max(np.abs(p2[3:] - p1[3:]))
        nt = int(np.ceil(td / ttol)) if td > ttol else 0
        nr = int(np.ceil(rd / rtol)) if rd > rtol else 0
        n = max(1, nt, nr)
        for j in range(1, n):
            out.append(p1 + (j / n) * (p2 - p1))
        out.append(p2)
    return np.array(out)

@njit(cache=True, inline='always')
def inside(x, y, z):
    """Exactly the notebook's are_points_in_corridor (inclusive bounds)."""
    a = (x <= 0) and (y >= 0) and (y <= W) and (z >= 0) and (z <= W)
    b = (x >= 0) and (x <= W) and (y >= 0) and (y <= YT) and (z >= 0) and (z <= W)
    c = (x >= 0) and (x <= W) and (y >= YT) and (y <= YT + W) and (z >= 0)
    return a or b or c
