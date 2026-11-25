"""Score a pose file with Google DeepMind's own, unmodified scoring code (3D moving sofa, snake corridor).

usage (from the release/ folder):
    python scoring/score_dm.py <pose file .npy|.json> <grid> [--end-pose-from-file] [--control-tang]
                               [--batch N] [--name NAME] [--allow-cpu]

  grid                  grid spacing of the initial sofa points (the published records use 0.001)
  --end-pose-from-file  use the pose file's last row as the appended end pose instead of DeepMind's fixed
                        (0, 6, 8, 90, 90, 90). This is the single disclosed change, used only for the double screw.
  --control-tang        first download Yuzhe Tang's published record path from his public gist (hash-checked)
                        and score it at the same grid, as a control. His data is not stored in this repository.
  --batch N             points per batch (default 5e7, as in DeepMind's code; does not change the result)
  --allow-cpu           run on CPU JAX (DeepMind's code is meant for a GPU; for quick checks at coarse grids)

Tolerances are fixed at 0.005 (translation) and 0.05 degrees (rotation), as in the published records.
The first and last rows of the pose file are dropped before scoring because DeepMind's code appends its own fixed
start and end poses. Writes results_<name>_<grid>.json in the current directory.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dm_common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pose_file')
    ap.add_argument('grid', type=float)
    ap.add_argument('--end-pose-from-file', action='store_true')
    ap.add_argument('--control-tang', action='store_true')
    ap.add_argument('--batch', type=float, default=5e7)
    ap.add_argument('--name', default=None)
    ap.add_argument('--allow-cpu', action='store_true')
    a = ap.parse_args()
    batch = int(a.batch)
    name = a.name or os.path.splitext(os.path.basename(a.pose_file))[0]

    RESULTS = C.check_device(a.allow_cpu)
    RESULTS['runs'] = []
    ns = C.load_dm()

    jobs = []
    if a.control_tang:
        T = C.fetch_tang()
        C.check_endpoints(T, False)
        jobs.append(('tang_w6_night3', T, False, 'Tang gist w6_night3_path.json'))
    P = C.load_pose_file(a.pose_file)
    C.check_endpoints(P, a.end_pose_from_file)
    jobs.append((name, P, a.end_pose_from_file, a.pose_file))

    out = 'results_%s_%g.json' % (name, a.grid)
    for nm, Q, own_end, src in jobs:
        print('%s: %s  poses %d  array sha256 %s  first %s  last %s' % (nm, src, len(Q), C.array_sha256(Q), Q[0].tolist(), Q[-1].tolist()))
        C.install_end_pose(ns, Q[-1] if own_end else None)
        C.print_min_step(ns, nm, Q)                       # float32 step-size guard
        t0 = time.time()
        mask, vol = ns['generate_sofa_and_evaluate_path'](Q[1:-1], a.grid, C.TRANSLATION_TOL, C.ROTATION_TOL, max_batch_size=batch)
        r = dict(path=nm, source=src, pose_array_sha256=C.array_sha256(Q), grid=a.grid,
                 translation_tol=C.TRANSLATION_TOL, rotation_tol_deg=C.ROTATION_TOL,
                 end_pose_from_file=bool(own_end), end_pose_used=(Q[-1].tolist() if own_end else C.DM_END),
                 volume=float(vol), survivors=int(np.asarray(mask).sum()), seconds=time.time() - t0)
        C.install_end_pose(ns, None)
        RESULTS['runs'].append(r)
        RESULTS['dm_notebook_sha256'] = C.DM_SHA256
        json.dump(RESULTS, open(out, 'w'), indent=1)
        note = '  [END POSE CHANGED: %s]' % Q[-1].tolist() if own_end else ''
        print('\n#### RESULT %s grid %g volume %.7f survivors %d (%.0f s)%s\n' % (nm, a.grid, r['volume'], r['survivors'], r['seconds'], note), flush=True)

    print('==== SUMMARY ====')
    for r in RESULTS['runs']:
        print('%-16s grid %-6g volume %.7f  survivors %d  %.0fs%s' % (r['path'], r['grid'], r['volume'], r['survivors'], r['seconds'],
                                                                     '  (end pose from file)' if r['end_pose_from_file'] else ''))
    print('results written to', out)


if __name__ == '__main__':
    main()
