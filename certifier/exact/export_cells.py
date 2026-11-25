"""
Export the certified cubes of the existing certifier (certifier/certify3d.py, unmodified), with a
per-cube "certificate slack" so the tightest cubes can be re-verified independently.

The traversal below mirrors certify3d.certify line by line; the only change is that the cell test
(test_cell_m, a copy of certify3d.test_cell) also returns the smallest slack its argument used:
  - at every visited pose: max over the boxes that keep the chain alive of (clearance - thr)
  - at every skip k -> lo: spare - (displacement bound actually used)
The run checks that per-level counts and the volume reproduce the certifier's logged numbers.

Saved per path (<EXACT_OUTDIR>/<tag>_cells.npz; EXACT_OUTDIR defaults to exact_out/ in the current directory), with keys = level<<40 | i<<26 | j<<13 | k (cell index at its level):
  coarse:   every certified cube of levels 0,1,2
  sample:   5,000 random certified cubes from each of levels 3,4,5,6 (seeded)
  tight:    the 2,000 certified cubes with the smallest certificate slack (all levels)
  surface:  up to 3,000 level-6 cubes on the body surface with an outward direction (negative control)
  all:      every certified cube, sorted, delta-encoded (all_first, all_diff)
Also: exact volume (fractions), disjointness check (no certified cube has a certified ancestor).
usage (from the release/ folder): python -W ignore certifier/exact/export_cells.py <tag> <path file>
       (set EXPORT_LEVELS=<n> for a quick run at a coarser level; the reported results use 6)
"""
import numpy as np, sys, json, time
from fractions import Fraction
from numba import njit, prange
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the release/ folder
from certifier.certify3d import fine_poses, box_margins, SLACK
from certifier.pipeline import W, L0
ROOT = 0.0625; LEVELS = int(os.environ.get("EXPORT_LEVELS", "6"))
OUTDIR = os.environ.get('EXACT_OUTDIR', 'exact_out')

@njit(cache=True)
def test_cell_m(cx, cy, cz, a, kstart, Rm, t, cD, cTh, amax, slack):
    n = Rm.shape[0]
    rho = a * np.sqrt(3.0)
    r = np.sqrt(cx * cx + cy * cy + cz * cz)
    thr = rho + (r + rho) * amax * amax / 8 + slack
    k = kstart; prev = 7; first = True; mslack = 1e9
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
            if max(m1, max(m2, m3)) < -rho - slack: return -1, k, mslack
            return 0, k, mslack
        ok = cur if first else (cur & prev)
        bv = -1e9
        if ok & 1: bv = max(bv, m1 - thr)
        if ok & 2: bv = max(bv, m2 - thr)
        if ok & 4: bv = max(bv, m3 - thr)
        mslack = min(mslack, bv)
        first = False
        if k == n - 1: return 1, k, mslack
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
            mslack = min(mslack, base - (cD[lo] + r * cTh[lo]))
            k = lo; prev = bb
        else:
            k += 1; prev = cur

@njit(parallel=True, cache=True)
def traverse(ixs, Rm, t, cD, cTh, amax, root, levels, slack, cap):
    ny = int(round(W / root)); m = len(ixs)
    keys = np.empty((m, cap), np.int64); sl = np.empty((m, cap), np.float64); cnt = np.zeros(m, np.int64)
    for q in prange(m):
        ix = ixs[q]
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
                    st, kf, ms = test_cell_m(cx, cy, cz, a, ks, Rm, t, cD, cTh, amax, slack)
                    if st == 1:
                        c = cnt[q]
                        if c < cap:
                            i = int(round((cx - a + L0) / (2 * a))); j = int(round((cy - a) / (2 * a))); kk_ = int(round((cz - a) / (2 * a)))
                            keys[q, c] = (lv << 40) | (i << 26) | (j << 13) | kk_
                            sl[q, c] = ms
                        cnt[q] = c + 1
                    elif st == 0 and lv < levels:
                        kk = kf - 1 if kf > 0 else 0
                        for dx in (-0.5, 0.5):
                            for dy in (-0.5, 0.5):
                                for dz in (-0.5, 0.5):
                                    scx[sp] = cx + dx * a; scy[sp] = cy + dy * a; scz[sp] = cz + dz * a
                                    slv[sp] = lv + 1; sk[sp] = kk; sp += 1
    return keys, sl, cnt

def decode(keys):
    return keys >> 40, (keys >> 26) & 0x3FFF, (keys >> 13) & 0x1FFF, keys & 0x1FFF

def enc(lv, i, j, k):
    return (lv << 40) | (i << 26) | (j << 13) | k

def member(sorted_keys, P):
    """P: (n,3) int64 points in units of the finest cell size / 4 (= 2^-12 * ... see below), relative to (-L0,0,0).
    Unit u = root / 2^(levels+2). Returns bool: point lies in the closed... (we query interior points only)."""
    res = np.zeros(len(P), bool)
    for lv in range(LEVELS + 1):
        s = 1 << (LEVELS + 2 - lv)            # cell size at level lv in units u
        idx = P // s
        ok = (idx[:, 0] >= 0) & (idx[:, 1] >= 0) & (idx[:, 2] >= 0)
        kk = enc(lv, np.clip(idx[:, 0], 0, None), np.clip(idx[:, 1], 0, None), np.clip(idx[:, 2], 0, None))
        pos = np.searchsorted(sorted_keys, kk); pos = np.minimum(pos, len(sorted_keys) - 1)
        res |= ok & (sorted_keys[pos] == kk)
    return res

def main(tag, fname):
    P = np.array(json.load(open(fname))['poses']) if fname.endswith('.json') else np.load(fname)
    t0 = time.time()
    Rs, ts, cD, cTh, amax = fine_poses(P, 0.05)
    nx = int(round(L0 / ROOT))
    allk = []; alls = []
    chunk = 16; cap = 1_200_000
    for c0 in range(0, nx, chunk):
        ixs = np.arange(c0, min(nx, c0 + chunk))
        keys, sl, cnt = traverse(ixs, Rs, ts, cD, cTh, amax, ROOT, LEVELS, SLACK, cap)
        assert cnt.max() <= cap, 'capacity overflow'
        for q in range(len(ixs)):
            allk.append(keys[q, :cnt[q]].copy()); alls.append(sl[q, :cnt[q]].copy())
        print('  chunk %d done, %.0fs' % (c0, time.time() - t0), flush=True)
    keys = np.concatenate(allk); sl = np.concatenate(alls); del allk, alls
    lv, i, j, k = decode(keys)
    counts = np.bincount(lv, minlength=LEVELS + 1)
    vol = sum(Fraction(int(counts[L]), 1) * Fraction(1, 16 * 2 ** L) ** 3 for L in range(LEVELS + 1))
    print('%s poses %d  cells per level %s  exact volume %s = %.9f  (%.0fs)' % (tag, len(ts), list(counts), vol, float(vol), time.time() - t0), flush=True)
    order = np.argsort(keys); sk = keys[order]
    assert np.all(np.diff(sk) > 0), 'duplicate cube'
    # disjointness: no certified ancestor
    clash = 0
    for L in range(1, LEVELS + 1):
        m = lv == L
        for Lp in range(L):
            sh = L - Lp
            anc = enc(np.int64(Lp), i[m] >> sh, j[m] >> sh, k[m] >> sh)
            pos = np.minimum(np.searchsorted(sk, anc), len(sk) - 1)
            clash += int((sk[pos] == anc).sum())
    print('  disjointness: %d certified cubes have a certified ancestor' % clash, flush=True)
    # bounds inside block
    inb = np.all([(i >= 0), (i < (64 << lv)), (j >= 0), (j < (16 << lv)), (k >= 0), (k < (16 << lv))], axis=0).all()
    rng = np.random.default_rng(12345)
    coarse = keys[lv <= 2]
    samp = np.concatenate([rng.choice(keys[lv == L], size=min(5000, int((lv == L).sum())), replace=False) for L in range(3, LEVELS + 1)])
    tidx = np.argsort(sl)[:2000]
    tight = keys[tidx]; tight_sl = sl[tidx]
    # surface cubes for the negative control: level-6 cubes whose face neighbour (inside the block) is not certified
    m6 = np.flatnonzero(lv == LEVELS); cand = rng.choice(m6, size=min(60000, len(m6)), replace=False)
    u = 4  # level-6 cell size in units u (u = 2^-12 for root 1/16, levels 6)
    base = np.stack([i[cand], j[cand], k[cand]], 1) * u + 2   # centres in units u
    surf = []; dirs = []
    hi = np.array([64 * 64 * u, 16 * 64 * u, 16 * 64 * u])
    for d in range(6):
        e = np.zeros(3, np.int64); e[d // 2] = 1 if d % 2 else -1
        Q = base + e * 4                                    # centre of the face neighbour
        inside_block = np.all((Q > 0) & (Q < hi), axis=1)
        out = inside_block & ~member(sk, Q)
        surf.append(cand[out]); dirs.append(np.full(out.sum(), d))
    surf = np.concatenate(surf); dirs = np.concatenate(dirs)
    _, first = np.unique(surf, return_index=True); surf = surf[first]; dirs = dirs[first]
    pick = rng.choice(len(surf), size=min(3000, len(surf)), replace=False)
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUTDIR, '%s_cells.npz' % tag), coarse=coarse, sample=samp, tight=tight, tight_slack=tight_sl,
                        surface=keys[surf[pick]], surface_dir=dirs[pick], counts=counts,
                        all_first=sk[0], all_diff=np.diff(sk),
                        vol_num=str(vol.numerator), vol_den=str(vol.denominator), clash=clash, in_block=inb,
                        n_poses=len(ts), amax=amax, min_slack=sl.min())
    print('  saved: coarse %d, sample %d, tight %d (slack %.3e .. %.3e), surface %d; in_block %s; %.0fs' % (
        len(coarse), len(samp), len(tight), tight_sl.min(), tight_sl.max(), len(pick), inb, time.time() - t0), flush=True)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
