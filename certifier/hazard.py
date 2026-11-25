"""Scan a path for rotation sub-steps too small for single precision (the DeepMind code NaNs on them on a GPU).
After the scorer's smoothing, every step's TRUE rotation angle (float64) should be >= thr degrees.
usage: hazard.py <paths...>"""
import sys, json, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the release/ folder
from scipy.spatial.transform import Rotation
from certifier.pipeline import smooth_path, add_start_and_end_poses

def scan(P, thr=1e-3, ttol=0.005, rtol=0.05):
    sm = smooth_path(add_start_and_end_poses(P[1:-1]), ttol, rtol)
    R = Rotation.from_euler('zyx', sm[:, 3:], degrees=True)
    a = np.degrees((R[1:] * R[:-1].inv()).magnitude())
    return sm, a

if __name__ == '__main__':
    for name in sys.argv[1:]:
        P = np.array(json.load(open(name))['poses']) if name.endswith('.json') else np.load(name)
        sm, a = scan(P)
        bad = np.where(a < 1e-3)[0]
        print('%s: smoothed %d  min true step %.3e deg  steps <1e-3: %d  <1e-4: %d  <1e-5: %d' % (name, len(sm), a.min(), len(bad), (a < 1e-4).sum(), (a < 1e-5).sum()))
        if len(bad):
            # group into runs
            runs = np.split(bad, np.where(np.diff(bad) > 1)[0] + 1)
            for r in runs[:12]:
                i, j = r[0], r[-1] + 1
                print('   run %d..%d (%d steps)  pos %s -> %s  euler %s -> %s' % (i, j, len(r), np.round(sm[i, :3], 3), np.round(sm[j, :3], 3), np.round(sm[i, 3:], 3), np.round(sm[j, 3:], 3)))
