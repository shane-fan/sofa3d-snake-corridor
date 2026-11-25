"""Shared helpers for score_dm.py and score_dm_split.py.

The scoring code is Google DeepMind's own: the notebook `the_3d_moving_sofa.ipynb` from
github.com/google-deepmind/alphaevolve_repository_of_problems is downloaded from a pinned commit, its SHA-256 is
checked, and the code cells listed in WANT are executed exactly as published (no edits) into a namespace.
Nothing in this file changes how a pose list is scored.

The only optional deviation is `--end-pose-from-file` (see install_end_pose): DeepMind's
`add_start_and_end_poses` appends the fixed end pose (0, 6, 8, 90, 90, 90). With the flag, the appended end pose is
the last row of the pose file instead. This was used for one path only (the double screw, end pose
(0, 5, 9.2, 90, 180, 90)) and is disclosed wherever that number is reported.
"""
import subprocess, json, hashlib, urllib.request
import numpy as np

# DeepMind's notebook at a pinned commit; the scoring cells are executed exactly as published.
DM_URL = ('https://raw.githubusercontent.com/google-deepmind/alphaevolve_repository_of_problems/'
          '8f447457957deac61e28bf1676746f0753b3b2f8/experiments/the_3d_moving_sofa/the_3d_moving_sofa.ipynb')
DM_SHA256 = 'a5b03434f3848bedcec5d47f87182660830209d811b4d31d3058357d3eabd73e'
WANT = ['# @title Imports', '# @title Corridor', '# @title Initial grid of sofa points', '# @title Path smoothing',
        '# @title Pose survival', '# @title Rotation step survival', '# @title Path survival', '# @title Volume calculation']

# Optional control: Yuzhe Tang's published record path, fetched at run time from his public gist (not stored here).
TANG_URL = ('https://gist.githubusercontent.com/Lomnus-ai/0208efd10e956eec39b5b2f8732613d5/raw/'
            '4ad4876ff1e05419fdac6a0a0004dd7f68c76a52/w6_night3_path.json')
TANG_SHA256 = '4707af40441f8df0a45d296179ec4774473959bc32d5e4568ac4310a7f372398'

TRANSLATION_TOL = 0.005   # same tolerances as the published records
ROTATION_TOL = 0.05       # degrees
DM_START = [-3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DM_END = [0.0, 6.0, 8.0, 90.0, 90.0, 90.0]


def check_device(allow_cpu):
    """Print the GPU and JAX device. DeepMind's code is meant for a GPU; on CPU it runs only with --allow-cpu."""
    try:
        print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'],
                             capture_output=True, text=True).stdout)
    except FileNotFoundError:
        print('nvidia-smi not found')
    import jax
    print('jax', jax.__version__, jax.devices())
    if jax.devices()[0].platform != 'gpu':
        assert allow_cpu, 'JAX does not see a GPU (pass --allow-cpu to run on CPU anyway)'
        print('WARNING: running on CPU (--allow-cpu). The reported numbers were computed on a GPU; '
              'CPU results can differ (about 0.1% at grid 0.02 in our test).')
    return {'jax': jax.__version__, 'device': str(jax.devices()[0])}


def load_dm():
    """Download DeepMind's notebook, check its hash, execute the scoring cells unmodified. Returns their namespace."""
    raw = urllib.request.urlopen(DM_URL).read()
    assert hashlib.sha256(raw).hexdigest() == DM_SHA256, 'DeepMind notebook differs from the copy we checked'
    srcs = [''.join(c['source']) for c in json.loads(raw)['cells'] if c['cell_type'] == 'code']
    ns = {'__name__': 'deepmind_notebook'}
    for w in WANT:
        s = [x for x in srcs if x.startswith(w)]
        assert len(s) == 1, w
        exec(s[0], ns)
    ns['_dm_add_start_and_end_poses'] = ns['add_start_and_end_poses']
    print('DeepMind scoring code loaded, unmodified (notebook sha256 %s)' % DM_SHA256)
    return ns


def load_pose_file(fname):
    """A pose file is a float64 array (.npy) or JSON (a list of rows, or an object with key "poses")."""
    if fname.endswith('.json'):
        d = json.load(open(fname))
        return np.array(d['poses'] if isinstance(d, dict) else d, dtype=np.float64)
    return np.load(fname).astype(np.float64)


def fetch_tang():
    """Yuzhe Tang's record path, downloaded from his public gist and hash-checked. Used only as a control."""
    raw = urllib.request.urlopen(TANG_URL).read()
    assert hashlib.sha256(raw).hexdigest() == TANG_SHA256, 'Tang path differs from the copy we checked'
    return np.array(json.loads(raw)['poses'], dtype=np.float64)


def array_sha256(P):
    return hashlib.sha256(np.ascontiguousarray(P, dtype=np.float64).tobytes()).hexdigest()


def check_endpoints(P, end_pose_from_file):
    """DeepMind's code appends its own start and end poses, so the first and last rows of the file are dropped.
    Check that the file agrees with what gets appended."""
    assert np.allclose(P[0], DM_START), 'first pose must be %s' % DM_START
    if not end_pose_from_file:
        assert np.allclose(P[-1], DM_END), ('last pose %s differs from the fixed end pose %s; '
                                             'pass --end-pose-from-file to use the file\'s own end pose' % (P[-1], DM_END))


def install_end_pose(ns, end_pose):
    """end_pose None: DeepMind's own add_start_and_end_poses (fixed end pose 0,6,8,90,90,90).
    Otherwise: THE SINGLE DISCLOSED CHANGE. The appended end pose is `end_pose` (the pose file's last row).
    Everything else (start pose, smoothing, survival test, volume rule) is DeepMind's code, unchanged.
    DeepMind's generate_sofa_and_evaluate_path looks the function up in its namespace at call time."""
    if end_pose is None:
        ns['add_start_and_end_poses'] = ns['_dm_add_start_and_end_poses']
        return
    e = np.array(end_pose, dtype=np.float64).copy()

    def add_start_and_end_poses(path):
        return np.vstack([[-3.0, 0.0, 0.0, 0.0, 0.0, 0.0], path, e])
    ns['add_start_and_end_poses'] = add_start_and_end_poses
    print('*** DISCLOSED CHANGE: the appended end pose is taken from the pose file: %s '
          '(DeepMind\'s fixed end pose is %s). All other scoring code is unmodified. ***' % (e.tolist(), DM_END))


def min_step(ns, P):
    """Guard: DeepMind's code returns NaN (volume 0) if a rotation sub-step is too small for single precision.
    Prints the true rotation per smoothed step, in degrees (Tang's record path, which scores correctly, has 7.5e-05)."""
    from scipy.spatial.transform import Rotation as _Rot
    sm = np.asarray(ns['smooth_path_to_tolerances'](ns['add_start_and_end_poses'](P[1:-1]), TRANSLATION_TOL, ROTATION_TOL))
    R = _Rot.from_euler('zyx', sm[:, 3:], degrees=True)
    a = np.degrees((R[1:] * R[:-1].inv()).magnitude())
    return len(sm), a.min(), int((a < 1e-5).sum())


def print_min_step(ns, name, P):
    print('%-16s smoothed %6d  min true rotation step %.2e deg  steps below 1e-5: %d' % ((name,) + min_step(ns, P)))
