"""Independent connectivity check of a certified cube set, working directly on the octree cells (no voxel raster,
no code from certify3d). Face adjacency only. Input: sorted int64 keys  level<<40 | i<<26 | j<<13 | k.
usage: conn_octree.py <tag> <cells.npz from export_cells.py | keys.npy>"""
import sys, time, numpy as np
from fractions import Fraction as F
from numba import njit

@njit(cache=True)
def bfind(keys, lo, hi, key):
    while lo < hi:
        mid = (lo + hi) >> 1
        if keys[mid] < key: lo = mid + 1
        else: hi = mid
    return lo

@njit(cache=True)
def find(par, x):
    r = x
    while par[r] != r: r = par[r]
    while par[x] != r:
        nx = par[x]; par[x] = r; x = nx
    return r

@njit(cache=True)
def components(keys, start, maxlev):
    n = keys.shape[0]
    par = np.arange(n).astype(np.int32)
    nadj = 0; nover = 0
    for c in range(n):
        key = keys[c]
        L = key >> 40; i = (key >> 26) & 0x3FFF; j = (key >> 13) & 0x1FFF; k = key & 0x1FFF
        nx = 64 << L; ny = 16 << L
        # disjointness: no certified ancestor
        for Lp in range(L):
            sh = L - Lp
            kk = (Lp << 40) | ((i >> sh) << 26) | ((j >> sh) << 13) | (k >> sh)
            p = bfind(keys, start[Lp], start[Lp + 1], kk)
            if p < start[Lp + 1] and keys[p] == kk: nover += 1
        for d in range(6):
            ni = i; nj = j; nk = k
            if d == 0: ni = i - 1
            elif d == 1: ni = i + 1
            elif d == 2: nj = j - 1
            elif d == 3: nj = j + 1
            elif d == 4: nk = k - 1
            else: nk = k + 1
            if ni < 0 or nj < 0 or nk < 0 or ni >= nx or nj >= ny or nk >= ny: continue
            for Lp in range(L, -1, -1):
                sh = L - Lp
                kk = (Lp << 40) | ((ni >> sh) << 26) | ((nj >> sh) << 13) | (nk >> sh)
                p = bfind(keys, start[Lp], start[Lp + 1], kk)
                if p < start[Lp + 1] and keys[p] == kk:
                    nadj += 1
                    a = find(par, c); b = find(par, p)
                    if a != b: par[a] = b
                    break
    # component weights in units of the finest cell volume
    w = np.zeros(n, np.int64); inblock = True
    for c in range(n):
        key = keys[c]; L = key >> 40
        i = (key >> 26) & 0x3FFF; j = (key >> 13) & 0x1FFF; k = key & 0x1FFF
        if i >= (64 << L) or j >= (16 << L) or k >= (16 << L): inblock = False
        w[find(par, c)] += np.int64(1) << (3 * (maxlev - L))
    return w, nadj, nover, inblock

def main(tag, fname):
    t0 = time.time()
    if fname.endswith('.npz'):
        D = np.load(fname); keys = np.concatenate([[D['all_first']], D['all_diff']]).cumsum().astype(np.int64)
    else:
        keys = np.load(fname).astype(np.int64)
    keys = np.sort(keys)
    assert np.all(np.diff(keys) > 0), 'duplicate keys'
    lv = keys >> 40; maxlev = int(lv.max())
    counts = np.bincount(lv, minlength=maxlev + 1)
    start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    vol = sum(F(int(counts[L]), (16 * 2 ** L) ** 3) for L in range(maxlev + 1))
    w, nadj, nover, inblock = components(keys, start, maxlev)
    comp = np.sort(w[w > 0])[::-1]
    unit = F(1, (16 * 2 ** maxlev) ** 3)
    print('%s: %d cubes, per level %s, volume %s = %.9f' % (tag, len(keys), counts.tolist(), vol, float(vol)))
    print('   face-connected components: %d ; largest = %s = %.9f ; others total %.3e (largest other %.3e) ; cubes with a certified ancestor: %d ; all inside block: %s ; adjacencies %d ; %.0fs' % (
        len(comp), F(int(comp[0])) * unit, float(F(int(comp[0])) * unit), float(F(int(comp[1:].sum())) * unit), float(F(int(comp[1])) * unit) if len(comp) > 1 else 0.0, nover, inblock, nadj, time.time() - t0), flush=True)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
