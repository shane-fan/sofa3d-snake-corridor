"""The same computation as DeepMind's generate_sofa_and_evaluate_path, with the batch loop spread over machines.

Only the loop is ours: point generation uses the notebook's own index formula (get_grid_dims_and_coords and the
unravel_index construction of generate_initial_sofa_points_in_batches), survival uses DeepMind's compute_path_survival,
and the volume uses DeepMind's compute_sofa_volume, all executed unmodified from the hash-checked notebook.
Splitting does not change the result: every point's survival is computed independently.

usage (from the release/ folder):
    python scoring/score_dm_split.py work    <tag> <pose file> <grid> <i0> <i1> [--end-pose-from-file] [--allow-cpu]
    python scoring/score_dm_split.py combine <tag> <pose file> <grid>           [--end-pose-from-file]

  work     computes batches i0 .. i1-1 (of 5e7 points each) and stores bit-packed masks in masks_<tag>/
           (batches already present are skipped, so a run can be resumed or spread over several machines;
           copy all masks_<tag>/b*.npz files to one place before combining)
  combine  loads every batch, checks they belong to the same pose file / grid / end-pose setting, and prints the
           volume and survivor count. It does not need a GPU.
  --end-pose-from-file, --batch, --allow-cpu: as in score_dm.py (the same values must be used for work and combine).
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dm_common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['work', 'combine'])
    ap.add_argument('tag')
    ap.add_argument('pose_file')
    ap.add_argument('grid', type=float)
    ap.add_argument('i0', type=int, nargs='?')
    ap.add_argument('i1', type=int, nargs='?')
    ap.add_argument('--end-pose-from-file', action='store_true')
    ap.add_argument('--batch', type=float, default=5e7)
    ap.add_argument('--allow-cpu', action='store_true')
    a = ap.parse_args()
    if a.mode == 'work':
        assert a.i0 is not None and a.i1 is not None, 'work needs <i0> <i1>'
        info = C.check_device(a.allow_cpu)
    else:
        import jax
        info = {'jax': jax.__version__, 'device': str(jax.devices()[0])}
    ns = C.load_dm()

    P = C.load_pose_file(a.pose_file)
    C.check_endpoints(P, a.end_pose_from_file)
    sha = C.array_sha256(P)
    print('path file', os.path.basename(a.pose_file), 'sha256', sha, 'end pose',
          P[-1] if a.end_pose_from_file else np.array(C.DM_END))
    C.install_end_pose(ns, P[-1] if a.end_pose_from_file else None)

    BATCH = int(a.batch)
    grid = a.grid
    dims, (xc, yc, zc) = ns['get_grid_dims_and_coords'](grid)
    total = dims[0] * dims[1] * dims[2]
    starts = list(range(0, total, BATCH)); nb = len(starts)
    mdir = 'masks_' + a.tag
    os.makedirs(mdir, exist_ok=True)
    meta = dict(path_sha256=sha, grid=grid, batch=BATCH, end_pose_from_file=bool(a.end_pose_from_file))

    if a.mode == 'work':
        C.print_min_step(ns, a.tag, P)                    # float32 step-size guard
        path = ns['add_start_and_end_poses'](P[1:-1])     # the scoring code adds the fixed start and end poses itself
        smoothed_path = ns['smooth_path_to_tolerances'](path, C.TRANSLATION_TOL, C.ROTATION_TOL)
        jnp = ns['jnp']
        for i in range(a.i0, min(a.i1, nb)):
            out = '%s/b%03d.npz' % (mdir, i)
            if os.path.exists(out): continue
            t0 = time.time()
            linear_indices = np.arange(starts[i], min(starts[i] + BATCH, total))
            ix, iy, iz = np.unravel_index(linear_indices, dims)
            batch_points = np.vstack([xc[ix], yc[iy], zc[iz]]).T
            m = np.array(ns['compute_path_survival'](jnp.array(smoothed_path), jnp.array(batch_points)))
            np.savez(out + '.tmp.npz', bits=np.packbits(m), n=m.size, alive=int(m.sum()), smoothed=len(smoothed_path),
                     meta=json.dumps(meta))
            os.replace(out + '.tmp.npz', out)
            print('BATCH %d/%d done: %d of %d alive, %.0f s' % (i, nb, int(m.sum()), m.size, time.time() - t0), flush=True)
        print('WORK DONE', a.tag, a.i0, a.i1, flush=True)
    else:
        parts = []
        for i in range(nb):
            d = np.load('%s/b%03d.npz' % (mdir, i))
            assert json.loads(str(d['meta'])) == meta, 'batch %d was computed with different settings' % i
            parts.append(np.unpackbits(d['bits'], count=int(d['n'])).astype(bool))
        mask = np.concatenate(parts); del parts
        assert mask.size == total
        alive = int(mask.sum())
        vol = ns['compute_sofa_volume'](mask.reshape(dims), (xc, yc, zc))
        note = '  [END POSE CHANGED: %s]' % P[-1].tolist() if a.end_pose_from_file else ''
        print('\n#### RESULT %s grid %g volume %.7f survivors %d (split driver, %d batches)%s\n' % (a.tag, grid, vol, alive, nb, note), flush=True)
        res = dict(info, tag=a.tag, source=a.pose_file, pose_array_sha256=sha, grid=grid, batches=nb,
                   translation_tol=C.TRANSLATION_TOL, rotation_tol_deg=C.ROTATION_TOL,
                   end_pose_from_file=bool(a.end_pose_from_file),
                   end_pose_used=(P[-1].tolist() if a.end_pose_from_file else C.DM_END),
                   volume=float(vol), survivors=alive, dm_notebook_sha256=C.DM_SHA256)
        json.dump(res, open('results_split_%s_%g.json' % (a.tag, grid), 'w'), indent=1)


if __name__ == '__main__':
    main()
