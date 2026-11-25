"""
Independent rigorous certifier for the 3D snake corridor (does not use DeepMind's point/cube rule).

Motion: consecutive poses (R_k, t_k) are joined by a screw step (rotation about a fixed axis through
the body origin at constant rate, plus straight-line translation), subdivided to small angles.

Shape: a union of axis-aligned cubes of the body frame (octree cells inside the block [-4,0]x[0,1]^2).
A cube with centre c and half-side a lies in the ball of radius rho = a*sqrt(3) around c, at every pose.
The corridor is the union of three convex boxes. For box b let m_b(q) be the L-infinity clearance of q.
A cube is certified if for every step k -> k+1 there is ONE box b with
        m_b(q_k) >= rho + s   and   m_b(q_{k+1}) >= rho + s,
where q_k is the centre at pose k and s >= |c| * angle^2 / 8 bounds how far the true screw path
of the centre leaves the chord q_k q_{k+1}.  (m_b is concave, so the whole chord has clearance
>= rho + s in box b, hence the true path has clearance >= rho, hence the whole cube is inside box b.)
Poses are skipped only when the clearance in one box exceeds the largest possible displacement.
All tests carry an extra absolute slack SLACK for floating-point rounding.
"""
import numpy as np, sys, time, json
from numba import njit, prange
from scipy.spatial.transform import Rotation
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the release/ folder
from certifier.pipeline import W, YT, L0

SLACK = 1e-9

def fine_poses(P, max_rot_deg=0.05, max_tr=0.01):
    P = np.asarray(P, float)
    R = Rotation.from_euler('zyx', P[:, 3:], degrees=True); t = P[:, :3]
    rel = R[1:] * R[:-1].inv(); rv = rel.as_rotvec(); ang = np.linalg.norm(rv, axis=1)
    dt = np.linalg.norm(np.diff(t, axis=0), axis=1)
    ns = np.maximum(1, np.ceil(np.maximum(ang / np.deg2rad(max_rot_deg), dt / max_tr))).astype(int)
    seg = np.repeat(np.arange(len(P) - 1), ns); lam = np.concatenate([np.arange(k) / k for k in ns])
    Rs = np.concatenate([(Rotation.from_rotvec(rv[seg] * lam[:, None]) * R[seg]).as_matrix(), R[-1:].as_matrix()])
    ts = np.vstack([t[seg] + lam[:, None] * (t[seg + 1] - t[seg]), t[-1:]])
    step_ang = np.append(np.repeat(ang / ns, ns), 0.0) * (1 + 1e-9) + 1e-12   # angle of step k -> k+1
    step_tr = np.append(np.repeat(dt / ns, ns), 0.0) * (1 + 1e-9) + 1e-12
    cD = np.concatenate([[0], np.cumsum(step_tr[:-1])]); cTh = np.concatenate([[0], np.cumsum(step_ang[:-1])])
    return Rs, ts, cD, cTh, step_ang.max()

@njit(cache=True, inline='always')
def box_margins(x, y, z):
    mz = min(z, W - z)
    m1 = min(min(W - x, min(y, W - y)), mz)
    m2 = min(min(min(x, W - x), min(y, YT + W - y)), mz)
    m3 = min(min(min(x, W - x), min(y - YT, YT + W - y)), z)
    return m1, m2, m3

@njit(cache=True)
def test_cell(cx, cy, cz, a, kstart, Rm, t, cD, cTh, amax, slack):
    """returns (status, k): status 1 certified, 0 failed at pose k (maybe partly inside), -1 entirely outside at pose k"""
    n = Rm.shape[0]
    rho = a * np.sqrt(3.0)
    r = np.sqrt(cx * cx + cy * cy + cz * cz)
    thr = rho + (r + rho) * amax * amax / 8 + slack
    k = kstart; prev = 7; first = True
    while True:
        R = Rm[k]
        x = R[0, 0] * cx + R[0, 1] * cy + R[0, 2] * cz + t[k, 0]
        y = R[1, 0] * cx + R[1, 1] * cy + R[1, 2] * cz + t[k, 1]
        z = R[2, 0] * cx + R[2, 1] * cy + R[2, 2] * cz + t[k, 2]
        m1, m2, m3 = box_margins(x, y, z)
        cur = 0
        if m1 >= thr: cur |= 1
        if m2 >= thr: cur |= 2
        if m3 >= thr: cur |= 4
        if (cur & prev) == 0 if not first else cur == 0:
            if max(m1, max(m2, m3)) < -rho - slack: return -1, k
            return 0, k
        first = False
        if k == n - 1: return 1, k
        # skip ahead inside the box with the most spare clearance
        best = m1; bb = 1
        if m2 > best: best = m2; bb = 2
        if m3 > best: best = m3; bb = 4
        spare = best - thr
        base = cD[k] + r * cTh[k] + spare
        if cD[k + 1] + r * cTh[k + 1] <= base:
            lo = k + 1; hi = n - 1
            if cD[hi] + r * cTh[hi] <= base: lo = hi
            else:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if cD[mid] + r * cTh[mid] <= base: lo = mid
                    else: hi = mid
            k = lo; prev = bb
        else:
            k += 1; prev = cur

@njit(parallel=True, cache=True)
def certify(Rm, t, cD, cTh, amax, root, levels, slack):
    nx = int(round(L0 / root)); ny = int(round(W / root))
    vol = np.zeros(nx); cnt = np.zeros((nx, levels + 1), np.int64)
    for ix in prange(nx):
        scx = np.empty(8 * levels + 16); scy = np.empty(8 * levels + 16); scz = np.empty(8 * levels + 16)
        slv = np.empty(8 * levels + 16, np.int64); sk = np.empty(8 * levels + 16, np.int64)
        for iy in range(ny):
            for iz in range(ny):
                sp = 0
                scx[0] = -L0 + (ix + 0.5) * root; scy[0] = (iy + 0.5) * root; scz[0] = (iz + 0.5) * root
                slv[0] = 0; sk[0] = 0; sp = 1
                while sp > 0:
                    sp -= 1
                    cx = scx[sp]; cy = scy[sp]; cz = scz[sp]; lv = slv[sp]; ks = sk[sp]
                    a = root / 2 / (1 << lv)
                    st, kf = test_cell(cx, cy, cz, a, ks, Rm, t, cD, cTh, amax, slack)
                    if st == 1:
                        vol[ix] += (2 * a) ** 3; cnt[ix, lv] += 1
                    elif st == 0 and lv < levels:
                        kk = kf - 1 if kf > 0 else 0
                        for dx in (-0.5, 0.5):
                            for dy in (-0.5, 0.5):
                                for dz in (-0.5, 0.5):
                                    scx[sp] = cx + dx * a; scy[sp] = cy + dy * a; scz[sp] = cz + dz * a
                                    slv[sp] = lv + 1; sk[sp] = kk; sp += 1
    return vol.sum(), cnt.sum(0)

def run(P, levels, root=0.0625, max_rot_deg=0.05, slack=SLACK):
    Rs, ts, cD, cTh, amax = fine_poses(P, max_rot_deg)
    v, cnt = certify(Rs, ts, cD, cTh, amax, root, levels, slack)
    return v, cnt, len(ts)

if __name__ == '__main__':
    levels = int(sys.argv[1])
    for name in sys.argv[2:]:
        P = np.array(json.load(open(name))['poses']) if name.endswith('.json') else np.load(name)
        t0 = time.time(); v, cnt, n = run(P, levels)
        print('%s  finest cell %.6f  poses %d  CERTIFIED VOLUME %.6f  cells per level %s  %.0fs' % (name, 0.0625 / 2 ** levels, n, v, list(cnt), time.time() - t0), flush=True)


# ---------------------------------------------------------------------------------------------
# Connected shape + controls
# ---------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def certify_raster(Rm, t, cD, cTh, amax, root, levels, slack):
    """Same as certify, but also writes the certified cells into a voxel grid at the finest level."""
    nx = int(round(L0 / root)); ny = int(round(W / root)); f = 1 << levels
    grid = np.zeros((nx * f, ny * f, ny * f), np.uint8)
    vol = np.zeros(nx)
    for ix in prange(nx):
        scx = np.empty(8 * levels + 16); scy = np.empty(8 * levels + 16); scz = np.empty(8 * levels + 16)
        slv = np.empty(8 * levels + 16, np.int64); sk = np.empty(8 * levels + 16, np.int64)
        for iy in range(ny):
            for iz in range(ny):
                scx[0] = -L0 + (ix + 0.5) * root; scy[0] = (iy + 0.5) * root; scz[0] = (iz + 0.5) * root
                slv[0] = 0; sk[0] = 0; sp = 1
                while sp > 0:
                    sp -= 1
                    cx = scx[sp]; cy = scy[sp]; cz = scz[sp]; lv = slv[sp]; ks = sk[sp]
                    a = root / 2 / (1 << lv)
                    st, kf = test_cell(cx, cy, cz, a, ks, Rm, t, cD, cTh, amax, slack)
                    if st == 1:
                        vol[ix] += (2 * a) ** 3
                        w = f >> lv; hf = root / f
                        i0 = int(round((cx - a + L0) / hf)); j0 = int(round((cy - a) / hf)); k0 = int(round((cz - a) / hf))
                        grid[i0:i0 + w, j0:j0 + w, k0:k0 + w] = 1
                    elif st == 0 and lv < levels:
                        kk = kf - 1 if kf > 0 else 0
                        for dx in (-0.5, 0.5):
                            for dy in (-0.5, 0.5):
                                for dz in (-0.5, 0.5):
                                    scx[sp] = cx + dx * a; scy[sp] = cy + dy * a; scz[sp] = cz + dz * a
                                    slv[sp] = lv + 1; sk[sp] = kk; sp += 1
    return vol.sum(), grid

@njit(cache=True)
def flood(grid, si, sj, sk_):
    """Mark with 2 the face-connected component of voxel (si,sj,sk_). Sweeps until stable."""
    nx, ny, nz = grid.shape
    grid[si, sj, sk_] = 2
    total = 1
    while True:
        changed = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if grid[i, j, k] == 1:
                        if ((i > 0 and grid[i - 1, j, k] == 2) or (j > 0 and grid[i, j - 1, k] == 2) or (k > 0 and grid[i, j, k - 1] == 2)
                                or (i < nx - 1 and grid[i + 1, j, k] == 2) or (j < ny - 1 and grid[i, j + 1, k] == 2) or (k < nz - 1 and grid[i, j, k + 1] == 2)):
                            grid[i, j, k] = 2; changed += 1
        for i in range(nx - 1, -1, -1):
            for j in range(ny - 1, -1, -1):
                for k in range(nz - 1, -1, -1):
                    if grid[i, j, k] == 1:
                        if ((i > 0 and grid[i - 1, j, k] == 2) or (j > 0 and grid[i, j - 1, k] == 2) or (k > 0 and grid[i, j, k - 1] == 2)
                                or (i < nx - 1 and grid[i + 1, j, k] == 2) or (j < ny - 1 and grid[i, j + 1, k] == 2) or (k < nz - 1 and grid[i, j, k + 1] == 2)):
                            grid[i, j, k] = 2; changed += 1
        total += changed
        if changed == 0: return total

@njit(parallel=True, cache=True)
def retest_scaled(grid, Rm, t, cD, cTh, amax, hf, scale, c0, stride, slack):
    """Negative control: re-test sampled voxels of the certified shape after scaling the shape about c0."""
    nx, ny, nz = grid.shape
    tested = np.zeros(nx, np.int64); passed = np.zeros(nx, np.int64)
    for i in prange(nx):
        if i % stride: continue
        for j in range(0, ny, stride):
            for k in range(0, nz, stride):
                if grid[i, j, k] == 2:
                    cx = c0[0] + scale * (-L0 + (i + 0.5) * hf - c0[0]); cy = c0[1] + scale * ((j + 0.5) * hf - c0[1]); cz = c0[2] + scale * ((k + 0.5) * hf - c0[2])
                    st, kf = test_cell(cx, cy, cz, scale * hf / 2, 0, Rm, t, cD, cTh, amax, slack)
                    tested[i] += 1; passed[i] += st == 1
    return tested.sum(), passed.sum()

def monte_carlo(grid, P, hf, n=20000, seed=0):
    """Independent brute-force spot check: random points of the certified shape, motion sampled 10x finer,
    tested against the corridor exactly as DeepMind's notebook defines it. Returns number of violations."""
    from certifier.pipeline import inside
    rng = np.random.default_rng(seed)
    idx = np.argwhere(grid[::7, ::7, ::7] == 2) * 7
    idx = idx[rng.choice(len(idx), size=min(n, len(idx)), replace=False)]
    pts = (idx + rng.random(idx.shape)) * hf + np.array([-L0, 0, 0])
    Rs, ts, _, _, _ = fine_poses(P, max_rot_deg=0.005, max_tr=0.001)
    bad = 0
    @njit(parallel=True)
    def chk(pts, Rs, ts):
        bad = np.zeros(len(pts), np.int64)
        for i in prange(len(pts)):
            for k in range(len(ts)):
                q = Rs[k] @ pts[i] + ts[k]
                if not inside(q[0], q[1], q[2]): bad[i] = 1; break
        return bad.sum()
    return int(chk(np.ascontiguousarray(pts), Rs, ts)), len(pts), len(ts)

def full_certificate(P, levels, root=0.0625, max_rot_deg=0.05, controls=True):
    Rs, ts, cD, cTh, amax = fine_poses(P, max_rot_deg)
    t0 = time.time()
    v, grid = certify_raster(Rs, ts, cD, cTh, amax, root, levels, SLACK)
    hf = root / (1 << levels)
    occ = np.argwhere(grid[::64, ::64, ::64] == 1)
    seed = occ[len(occ) // 2] * 64
    n2 = flood(grid, seed[0], seed[1], seed[2])
    out = dict(finest_cell=hf, poses=len(ts), certified_all=float(v), certified_connected=float(n2) * hf ** 3, seconds=time.time() - t0)
    if controls:
        c0 = np.array([-2.0, 0.5, 0.5])
        for sc in (1.0, 1.002, 1.01):
            te, pa = retest_scaled(grid, Rs, ts, cD, cTh, amax, hf, sc, c0, 5, SLACK)
            out['retest_scale_%g' % sc] = '%d of %d sampled cells pass' % (pa, te)
        out['monte_carlo_violations'] = '%d of %d random points (motion sampled at %d poses)' % monte_carlo(grid, P, hf)
    return out, grid
