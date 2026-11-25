"""Connectivity of the cube set counted by DeepMind's compute_sofa_volume, from the saved point-survival masks.
usage: python mask_components.py <masks_dir> <grid>
Rebuilds the cube mask exactly as compute_sofa_volume does (a cube counts iff all 64 points of its 4x4x4
neighbourhood survive), then labels face-connected components."""
import sys, glob, numpy as np, time
from scipy import ndimage
d, h = sys.argv[1], float(sys.argv[2])
nx, ny, nz = int(round(4 / h)) + 1, int(round(1 / h)) + 1, int(round(1 / h)) + 1
files = sorted(glob.glob(d + '/b*.npz')); parts = []
for f in files:
    z = np.load(f); parts.append(np.unpackbits(z['bits'], count=int(z['n'])).astype(bool))
m = np.concatenate(parts); del parts
assert m.size == nx * ny * nz, (m.size, nx, ny, nz)
print('points', m.size, 'alive', int(m.sum()), flush=True)
m = m.reshape(nx, ny, nz)
c = np.ones((nx - 3, ny - 3, nz - 3), dtype=bool)
for i in range(4):
    for j in range(4):
        for k in range(4):
            c &= m[i:nx - 3 + i, j:ny - 3 + j, k:nz - 3 + k]
del m
tot = int(c.sum()); print('cubes counted', tot, 'volume %.7f' % (tot * h ** 3), flush=True)
t0 = time.time(); lab, n = ndimage.label(c); del c
sizes = np.bincount(lab.ravel())[1:]; sizes.sort()
print('face-connected components', n, 'largest', int(sizes[-1]), 'largest volume %.7f' % (sizes[-1] * h ** 3),
      'cubes outside largest', tot - int(sizes[-1]), 'next sizes', sizes[-6:-1][::-1].tolist(), '%.0fs' % (time.time() - t0), flush=True)
