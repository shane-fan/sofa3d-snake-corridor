"""
Driver: exact re-verification of exported cubes + negative controls.
usage (from the release/ folder): python -W ignore certifier/exact/run_exact.py <tag> <path file> [--all]
Reads <EXACT_OUTDIR>/<tag>_cells.npz (from export_cells.py), writes <EXACT_OUTDIR>/<tag>_exact.json
(EXACT_OUTDIR defaults to exact_out/ in the current directory).
"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # exact_verify.py sits next to this file
OUTDIR = os.environ.get('EXACT_OUTDIR', 'exact_out')
import numpy as np
from fractions import Fraction as F
from exact_verify import prepare, thresholds, check_full, check_skip, keys_to_cubes, load_poses, bounds_ok, CONST, CU, PU, MU

def run_set(pose, C, A):
    assert bounds_ok(pose, C)
    T, E, r, rho = thresholds(C, A, pose)
    ok, marg, fail = check_full(pose['M'], pose['T'], C, T, CONST['W'], CONST['Y5'], CONST['Y6'])
    return ok, marg, fail

def main(tag, fname, do_all):
    t0 = time.time(); rep = dict(tag=tag, path=fname)
    D = np.load(os.path.join(OUTDIR, '%s_cells.npz' % tag))
    P = load_poses(fname)
    pose = prepare(P)
    # sanity: integer matrices at the original poses agree with a plain float evaluation of Rx Ry Rz
    def Rf(y, p, r):
        y, p, r = np.radians([y, p, r])
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        return Rx @ Ry @ Rz
    dev = abs(pose['M'][-1] / 2.0 ** MU - Rf(*P[-1, 3:])).max()
    rep.update(n_fine_poses=int(len(pose['T'])), delta=float(pose['delta']), theta_max_rad=float(pose['theta_max']),
               float_sanity_last_pose=float(dev))
    print('%s: %d fine poses, delta %.3e, theta_max %.4e, float sanity %.1e' % (tag, len(pose['T']), float(pose['delta']), float(pose['theta_max']), dev), flush=True)
    counts = D['counts']; vol = F(int(D['vol_num']), int(D['vol_den']))
    rep.update(cells_per_level=[int(c) for c in counts], exported_volume_exact='%s' % vol, exported_volume=float(vol),
               disjoint_clashes=int(D['clash']), in_block=bool(D['in_block']), certifier_min_slack=float(D['min_slack']))
    sets = {}
    for name in ('coarse', 'sample', 'tight', 'surface'):
        keys = D[name]; C, A, lv = keys_to_cubes(keys)
        ok, marg, fail = run_set(pose, C, A)
        per = {int(L): [int(ok[lv == L].sum()), int((lv == L).sum())] for L in np.unique(lv)}
        sets[name] = dict(passed=int(ok.sum()), total=int(len(ok)), per_level_pass_of_total=per,
                          min_exact_margin=float(marg.min()) / 2 ** PU, volume=float(((2 * A.astype(float) / 2 ** CU) ** 3).sum()))
        print('  %-8s %d / %d pass; per level %s; min exact margin %.3e' % (name, ok.sum(), len(ok), per, marg.min() / 2 ** PU), flush=True)
        if name == 'tight':
            sets[name]['certifier_slack_range'] = [float(D['tight_slack'].min()), float(D['tight_slack'].max())]
            sets[name]['exact_margin_of_tightest_10'] = [float(m) / 2 ** PU for m in np.sort(marg)[:10]]
    rep['sets'] = sets
    # ---------------- negative controls ----------------
    keys = D['surface']; dirs = D['surface_dir']; C, A, lv = keys_to_cubes(keys)
    sh = math.ceil(0.003 * 2 ** CU)                       # 197 units = 0.0030060
    e = np.zeros_like(C); e[np.arange(len(C)), dirs // 2] = np.where(dirs % 2 == 1, 1, -1)
    ok, marg, fail = run_set(pose, C + sh * e, A)
    rep['neg_shift_outward'] = dict(shift=sh / 2 ** CU, rejected=int((~ok).sum()), total=int(len(ok)))
    print('  NEG shift %.6f outward: %d / %d rejected' % (sh / 2 ** CU, (~ok).sum(), len(ok)), flush=True)
    for sc_num, sc_den in ((1002, 1000), (1010, 1000)):
        allk = np.concatenate([D['coarse'], D['sample'], D['surface']]); C, A, lv = keys_to_cubes(allk)
        c0 = np.array([-2 * 2 ** CU, 2 ** CU // 2, 2 ** CU // 2], np.int64)
        Cs = c0 + np.rint((C - c0) * (sc_num / sc_den)).astype(np.int64)     # centre rounded to 2^-16 (<= 7.6e-6 shift)
        As = -((-A * sc_num) // sc_den)                                         # ceil
        ok, marg, fail = run_set(pose, Cs, As)
        nsurf = len(D['surface'])
        rep['neg_scale_%g' % (sc_num / sc_den)] = dict(rejected=int((~ok).sum()), total=int(len(ok)),
                                                     surface_rejected=int((~ok[-nsurf:]).sum()), surface_total=nsurf)
        print('  NEG scale %.3f: %d / %d cubes rejected (surface cubes: %d / %d)' % (sc_num / sc_den, (~ok).sum(), len(ok), (~ok[-nsurf:]).sum(), nsurf), flush=True)
    # ---------------- every certified cube (exact skip variant) ----------------
    if do_all:
        sk = np.concatenate([[D['all_first']], D['all_diff']]).cumsum()
        C, A, lv = keys_to_cubes(sk)
        assert bounds_ok(pose, C)
        T, E, r, rho = thresholds(C, A, pose)
        assert int(r.max()) * int(pose['cth'][-1]) + int(pose['ctr'][-1]) < 2 ** 62
        t1 = time.time()
        ok, fail, ev = check_skip(pose['M'], pose['T'], C, T, r, rho, E, pose['ctr'], pose['cth'], CONST['W'], CONST['Y5'], CONST['Y6'])
        per = {int(L): [int(ok[lv == L].sum()), int((lv == L).sum())] for L in np.unique(lv)}
        volp = sum(F(int(ok[lv == L].sum())) * F(1, 16 * 2 ** int(L)) ** 3 for L in np.unique(lv))
        rep['all_cubes'] = dict(passed=int(ok.sum()), total=int(len(ok)), per_level_pass_of_total=per,
                                verified_volume_exact=str(volp), verified_volume=float(volp),
                                mean_pose_evals=float(ev.mean()), seconds=time.time() - t1)
        print('  ALL cubes (exact, skip variant): %d / %d pass, verified volume %.9f, mean %.0f pose evals/cube, %.0fs' % (
            ok.sum(), len(ok), float(volp), ev.mean(), time.time() - t1), flush=True)
        if not ok.all():
            bad = np.flatnonzero(~ok)[:50]
            Cb, Ab = C[bad], A[bad]
            okb, mb, fb = run_set(pose, Cb, Ab)
            rep['all_cubes']['failed_rechecked_without_skip'] = [int(okb.sum()), int(len(okb))]
            print('    first failures re-checked without skip: %d / %d pass' % (okb.sum(), len(okb)), flush=True)
    rep['seconds'] = time.time() - t0
    json.dump(rep, open(os.path.join(OUTDIR, '%s_exact.json' % tag), 'w'), indent=1)
    print('  done %.0fs' % (time.time() - t0), flush=True)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], '--all' in sys.argv)
