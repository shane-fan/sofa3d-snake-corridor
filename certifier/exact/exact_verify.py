"""
Independent exact re-verification of certified cubes for the 3D snake corridor.
Shares no code with certifier/certify3d.py (only numpy / numba / mpmath / fractions).

Motion (the object being certified): pose k = (x,y,z,yaw,pitch,roll) in degrees, read as binary64 numbers and
taken as exact rationals; R_k = Rx(roll) Ry(pitch) Rz(yaw) exactly (= scipy from_euler('zyx', ..., degrees=True));
between pose k and k+1 the rotation is the geodesic R(l) = exp(l*Omega_k) R_k (Omega_k = log(R_{k+1} R_k^T),
angle phi_k <= pi) and the translation is linear, l in [0,1].

Rigour:
  * every original pose is evaluated in mpmath interval arithmetic (outward rounding, 120-bit) as a unit quaternion;
    relative rotation, its angle phi_k (upper bound) and every fine pose l = i/n_k are interval enclosures;
  * each fine rotation is rounded to an integer matrix M/2^36; delta = rigorous bound on ||R_true - M/2^36||_F;
  * each fine translation is the exact rational t_k + (i/n_k)(t_{k+1}-t_k), rounded to a multiple of 2^-52;
  * cube centres are exact multiples of 2^-16; world centres M C + T are computed EXACTLY in int64 (units 2^-52)
    after an a-priori overflow check, and box clearances are exact integers;
  * acceptance of step j -> j+1: one maximal convex box b of the corridor with
        clearance_b(Q_j) >= T  and  clearance_b(Q_{j+1}) >= T,
        T = ceil( rho_up + r_up*(theta_max^2/8 + delta) + 2^-52 )   (all upper bounds, exact fractions)
    where r_up >= |c|, rho_up >= a*sqrt(3), theta_max >= every fine step angle.
  Proof: true centre at pose j is within r*delta + 2^-52 of Q_j; clearance is concave, so the chord between the
  true centres has clearance >= rho + r*theta^2/8 in box b; the true centre path leaves the chord by at most
  r*theta^2/8 (interpolation-error bound for s -> exp(s*theta*K)v); every cube point is within rho of the centre.
"""
import numpy as np, sys, json, math, time
from fractions import Fraction as F
from numba import njit, prange
import mpmath
from mpmath import iv

iv.prec = 120
CU = 16      # cube coordinates in units 2^-16
MU = 36      # rotation entries in units 2^-36
PU = CU + MU # positions in units 2^-52
MAX_ROT_DEG = 0.05; MAX_TR = 0.01  # subdivision rule (same discretisation as the certifier; any choice is sound)

def _raw(t):
    sign, man, exp, bc = t
    assert man != 0 or (exp == 0 and bc == 0), 'non-finite endpoint'
    v = F(int(man)) * (F(2) ** int(exp)) if man else F(0)
    return -v if sign else v

def lo_hi(x):
    """exact rational endpoints of an mpmath interval"""
    a, b = x._mpi_
    return _raw(a), _raw(b)

def ivf(x):
    return iv.mpf(float(x))   # binary64 -> exact interval [x,x]

def qmul(p, q):
    a1, b1, c1, d1 = p; a2, b2, c2, d2 = q
    return (a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2)

def euler_q(yaw, pitch, roll):
    h = iv.pi / 360
    Y, Pi, Rr = ivf(yaw) * h, ivf(pitch) * h, ivf(roll) * h
    z0 = iv.mpf(0)
    qz = (iv.cos(Y), z0, z0, iv.sin(Y)); qy = (iv.cos(Pi), z0, iv.sin(Pi), z0); qx = (iv.cos(Rr), iv.sin(Rr), z0, z0)
    return qmul(qx, qmul(qy, qz))      # R = Rx Ry Rz

def qmat(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]

def prepare(P, log=print):
    """Fine poses: integer rotation matrices (units 2^-MU), integer translations (units 2^-PU), rigorous bounds."""
    P = [[float(v) for v in row] for row in P]
    t0 = time.time()
    Q = [euler_q(p[3], p[4], p[5]) for p in P]
    Ms, Ts, delta, theta_max, max_phi, extra = [], [], F(0), F(0), 0.0, F(0)
    step_tr_up, step_th_up = [], []      # per fine step: |dt| upper (units 2^-PU), theta upper (units 2^-MU)
    S = 2 ** MU; SP = 2 ** PU
    def add_pose(Rint_iv, tfr):
        nonlocal delta
        Mi = []
        emax = F(0)
        for r in range(3):
            row = []
            for c in range(3):
                lo, hi = lo_hi(Rint_iv[r][c])
                assert hi - lo < F(1, 2 ** 80), 'interval too wide'
                m = round((lo + hi) / 2 * S); row.append(m)
                emax = max(emax, abs(F(m, S) - lo), abs(F(m, S) - hi))
            Mi.append(row)
        delta = max(delta, 3 * emax)
        Ms.append(Mi); Ts.append([round(v * SP) for v in tfr])
    for k in range(len(P) - 1):
        q0, q1 = Q[k], Q[k + 1]
        rel = qmul(q1, (q0[0], -q0[1], -q0[2], -q0[3]))
        if lo_hi(rel[0])[1] < 0: rel = tuple(-v for v in rel)
        assert lo_hi(rel[0])[0] > 0, 'relative rotation near pi'
        vn = iv.sqrt(rel[1] ** 2 + rel[2] ** 2 + rel[3] ** 2)
        half = iv.atan2(vn, rel[0])
        phi_up = 2 * lo_hi(half)[1]; max_phi = max(max_phi, float(phi_up))
        ta, tb = [F(v) for v in P[k][:3]], [F(v) for v in P[k + 1][:3]]
        dtv = [b - a for a, b in zip(ta, tb)]
        dt_f = math.sqrt(float(sum(d * d for d in dtv)))
        n = max(1, math.ceil(max(float(phi_up) / math.radians(MAX_ROT_DEG), dt_f / MAX_TR)))
        theta_max = max(theta_max, phi_up / n)
        tr2 = sum(d * d for d in dtv) / (n * n)               # exact |step translation|^2
        tr_up = math.isqrt(math.ceil(tr2 * SP * SP)) + 1       # >= |dt/n| * 2^PU
        th_up = math.ceil(phi_up / n * S) + 1
        tiny = lo_hi(vn)[1] < F(1, 2 ** 60)
        if not tiny:
            u = (rel[1] / vn, rel[2] / vn, rel[3] / vn)
        for i in range(n):
            lam = F(i, n)
            tfr = [a + lam * d for a, d in zip(ta, dtv)]
            if i == 0:
                q = q0
            elif tiny:
                # rotation angle < 2^-59: use R_k and add the whole angle to the error bound
                q = q0; extra = max(extra, 2 * phi_up)   # ||R(l)-R_k||_F <= 2*phi
            else:
                hl = half * iv.mpf(lam.numerator) / lam.denominator
                s = iv.sin(hl)
                q = qmul((iv.cos(hl), s * u[0], s * u[1], s * u[2]), q0)
            add_pose(qmat(q), tfr)
            step_tr_up.append(tr_up); step_th_up.append(th_up)
    add_pose(qmat(Q[-1]), [F(v) for v in P[-1][:3]])
    delta = delta + extra
    Mi = np.array(Ms, np.int64); Ti = np.array(Ts, np.int64)
    log('  prepared %d fine poses in %.0fs: delta <= %.3e, theta_max <= %.6e rad (%.5f deg), max segment angle %.4f rad' % (
        len(Ts), time.time() - t0, float(delta), float(theta_max), math.degrees(float(theta_max)), max_phi))
    return dict(M=Mi, T=Ti, delta=delta, theta_max=theta_max,
                ctr=np.concatenate([[0], np.cumsum(step_tr_up)]).astype(np.int64),
                cth=np.concatenate([[0], np.cumsum(step_th_up)]).astype(np.int64))

def isqrt_up(v):
    """elementwise integer upper bound of sqrt for int64 arrays < 2^52 (verified exactly)"""
    r = np.floor(np.sqrt(v.astype(np.float64))).astype(np.int64) + 1
    assert np.all(r * r >= v)
    return r

S3 = math.isqrt(3 * 2 ** (2 * MU)) + 1          # integer > sqrt(3) * 2^MU
assert S3 * S3 > 3 * 2 ** (2 * MU)

def thresholds(C, A, pose):
    """C (n,3) int (units 2^-CU), A (n,) int half-sides (units 2^-CU).
    Returns T (step threshold), E (pose error bound), r_up (units 2^-CU), rho_up (units 2^-PU).
    r_up only multiplies quantities <= 1e-6 (or path-length bounds in the skip), so 2^-16 resolution is harmless;
    rho = a*sqrt(3) is bounded at 2^-52 resolution: rho_up = A * S3 >= A*2^-16 * sqrt(3) * 2^52."""
    assert np.abs(C).max() < 2 ** 24 and A.max() < 2 ** 20
    r_up = isqrt_up((C * C).sum(1)); rho_up = A.astype(np.int64) * S3
    K = math.ceil((pose['theta_max'] ** 2 / 8 + pose['delta']) * 2 ** MU)     # upper bound, units 2^-MU
    Kd = math.ceil(pose['delta'] * 2 ** MU)
    T = rho_up + r_up * K + 1                  # >= rho + r*(theta^2/8 + delta) + 2^-PU, units 2^-PU
    E = r_up * Kd + 1                          # >= r*delta + 2^-PU
    return T.astype(np.int64), E.astype(np.int64), r_up, rho_up

@njit(cache=True, inline='always')
def clear3(x, y, z, W, Y5, Y6):
    m1 = min(min(W - x, y), min(W - y, min(z, W - z)))                       # {x<=1, 0<=y<=1, 0<=z<=1}
    m2 = min(min(min(x, W - x), min(y, Y6 - y)), min(z, W - z))               # {0<=x<=1, 0<=y<=6, 0<=z<=1}
    m3 = min(min(min(x, W - x), min(y - Y5, Y6 - y)), z)                      # {0<=x<=1, 5<=y<=6, z>=0}
    return m1, m2, m3

@njit(parallel=True, cache=True)
def check_full(M, T, C, thr, W, Y5, Y6):
    """Every fine step, no skipping. Returns ok flag, exact min margin (units 2^-PU), first failing step."""
    n = C.shape[0]; N = M.shape[0]
    ok = np.ones(n, np.bool_); marg = np.empty(n, np.int64); fail = np.full(n, -1, np.int64)
    for c in prange(n):
        c0 = C[c, 0]; c1 = C[c, 1]; c2 = C[c, 2]; th = thr[c]
        p1 = 0; p2 = 0; p3 = 0; mm = np.int64(1) << 62
        for j in range(N):
            x = M[j, 0, 0] * c0 + M[j, 0, 1] * c1 + M[j, 0, 2] * c2 + T[j, 0]
            y = M[j, 1, 0] * c0 + M[j, 1, 1] * c1 + M[j, 1, 2] * c2 + T[j, 1]
            z = M[j, 2, 0] * c0 + M[j, 2, 1] * c1 + M[j, 2, 2] * c2 + T[j, 2]
            m1, m2, m3 = clear3(x, y, z, W, Y5, Y6)
            if j > 0:
                best = max(min(m1, p1), max(min(m2, p2), min(m3, p3))) - th
                if best < mm: mm = best
                if best < 0 and fail[c] < 0:
                    fail[c] = j - 1; ok[c] = False
            p1 = m1; p2 = m2; p3 = m3
        marg[c] = mm
    return ok, marg, fail

@njit(cache=True)
def skip_one(c0, c1, c2, th, r, rho, E, M, T, ctr, cth, W, Y5, Y6):
    """returns (ok, failing step, pose evaluations)"""
    N = M.shape[0]
    j = 0; p1 = 0; p2 = 0; p3 = 0; have_prev = False; ev = 0
    while True:
        x = M[j, 0, 0] * c0 + M[j, 0, 1] * c1 + M[j, 0, 2] * c2 + T[j, 0]
        y = M[j, 1, 0] * c0 + M[j, 1, 1] * c1 + M[j, 1, 2] * c2 + T[j, 1]
        z = M[j, 2, 0] * c0 + M[j, 2, 1] * c1 + M[j, 2, 2] * c2 + T[j, 2]
        ev += 1
        m1, m2, m3 = clear3(x, y, z, W, Y5, Y6)
        if have_prev:
            if max(min(m1, p1), max(min(m2, p2), min(m3, p3))) < th:
                return False, j - 1, ev
        elif max(m1, max(m2, m3)) < th:          # start of a chain (pose 0, or after a skip)
            return False, j, ev
        if j == N - 1: return True, -1, ev
        m = max(m1, max(m2, m3))
        budget = m - E - rho                       # all in units 2^-52
        lo = j
        if budget > 0:
            hi = N - 1
            while hi > lo:
                mid = (lo + hi + 1) // 2
                d = ctr[mid] - ctr[j] + r * (cth[mid] - cth[j])   # path-length bound, units 2^-52
                if d <= budget: lo = mid
                else: hi = mid - 1
        if lo >= j + 2:
            j = lo; have_prev = False   # [j, lo] covered continuously in the box with clearance m
        else:
            j += 1; have_prev = True
        p1 = m1; p2 = m2; p3 = m3

@njit(parallel=True, cache=True)
def check_skip(M, T, C, thr, rup, rhoup, E, ctr, cth, W, Y5, Y6):
    """Same acceptance as check_full plus an exact skip: from pose j with clearance m (box b), every pose j' with
    path-length bound D(j,j') = ctr[j']-ctr[j] + r_up*(cth[j']-cth[j]) <= m - E - rho_up is covered: the true
    centre starts within E of Q_j and moves at most D, so it keeps clearance >= rho_up in box b throughout."""
    n = C.shape[0]
    ok = np.ones(n, np.bool_); fail = np.full(n, -1, np.int64); evals = np.zeros(n, np.int64)
    for c in prange(n):
        o, f, e = skip_one(C[c, 0], C[c, 1], C[c, 2], thr[c], rup[c], rhoup[c], E[c], M, T, ctr, cth, W, Y5, Y6)
        ok[c] = o; fail[c] = f; evals[c] = e
    return ok, fail, evals

def keys_to_cubes(keys, levels=6):
    lv = keys >> 40; i = (keys >> 26) & 0x3FFF; j = (keys >> 13) & 0x1FFF; k = keys & 0x1FFF
    # level-lv cell side = 2^-4 / 2^lv = 2^(12-lv) units of 2^-16; centre = corner + half
    side = np.left_shift(np.int64(1), (12 - lv).astype(np.int64))
    C = np.stack([i * side - 4 * 2 ** CU + side // 2, j * side + side // 2, k * side + side // 2], 1).astype(np.int64)
    return C, (side // 2).astype(np.int64), lv

def load_poses(fname):
    return np.array(json.load(open(fname))['poses'], float) if fname.endswith('.json') else np.load(fname)

def bounds_ok(pose, C):
    mM = int(np.abs(pose['M']).max()); mC = int(np.abs(C).max()); mT = int(np.abs(pose['T']).max())
    return 3 * mM * mC + mT < 2 ** 61

CONST = dict(W=np.int64(2 ** PU), Y5=np.int64(5 * 2 ** PU), Y6=np.int64(6 * 2 ** PU))
