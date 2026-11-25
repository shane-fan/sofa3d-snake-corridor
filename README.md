# 3D moving sofa, snake corridor: pose files and checks for new lower bounds

Author: Shane Fan
Paper: [link to be added]

This folder holds the motions (pose lists) behind new lower bounds for the 3D moving sofa problem in the "snake"
corridor, as posed in Google DeepMind's AlphaEvolve repository of problems. It also holds the code needed to check them
three ways:

1. **DeepMind's own scoring code, unmodified.** `scoring/` downloads DeepMind's notebook from a pinned commit, checks
   its SHA-256 and runs its scoring cells as published.
2. **An independent certifier.** `certifier/certify3d.py` certifies a union of cubes against the continuous motion,
   with rigorous error bounds. It does not use DeepMind's point-and-cube rule.
3. **An exact-arithmetic re-check.** `certifier/exact/` re-verifies every certified cube with exact integer and
   rational arithmetic and interval-arithmetic rotations, sharing no code with the certifier.

## Results

All scores use DeepMind's tolerances (0.005 translation, 0.05° rotation) at grid 0.001, the resolution of the
published records. The full table, with the log behind every number, is in [`results/RESULTS.md`](results/RESULTS.md).

**Result 1: path D2, fixed end pose, DeepMind's code unmodified.** `paths/D2.npy` ends at DeepMind's fixed end pose
(0, 6, 8, 90, 90, 90) and scores **1.8893629** (1,910,309,317 surviving grid points). The previous published record is
**1.8558404** (Yuzhe Tang). Run as a control in the same setup, his path gave exactly 1.8558404. The split and
unsplit runs of D2 agree exactly.

**Result 2: double screw, free end orientation.** `paths/double_screw.npy` ends at (0, 5, 9.2, 90, 180, 90) instead
of (0, 6, 8, 90, 90, 90). It scores **1.9011554** (1,922,601,669 points) with DeepMind's code, with one change: the
end pose that the code appends is the file's own last row instead of the fixed one. This is a different
(less constrained) problem from DeepMind's fixed-end-pose formulation, so this number is not a score for that
formulation. It should always be quoted with this caveat.

| | DeepMind's code, grid 0.001 | Certifier, level 6 | Exact re-check (level 6 cubes) |
|---|---|---|---|
| Tang's record path (control, not included) | 1.8558404 | 1.870832 | all pass |
| B | 1.8650145 | 1.880143 | all pass, 1.880143894 |
| D | 1.8870264 | 1.904259 | all pass, 1.904259124 |
| **D2** (Result 1) | **1.8893629** | **1.906472** | all 14,267,073 cubes pass, 1.906472175 |
| double screw (Result 2, end pose changed) | 1.9011554 | 1.918910 | all pass, 1.918910558 |

The certifier's level-7 values (finest cube 0.000488) are D2 1.911434 (the same on two machines), D 1.909281 and
double screw 1.923975. The certifier and
the exact re-check measure a different, continuous-motion notion of "fits", so their numbers are not directly
comparable with DeepMind's. They are an independent check that each motion really carries a body of roughly that
volume.

## Contents

```
paths/        D2, D, B, double_screw: .npy (float64) and .json (same numbers, text), SHA256SUMS, check_paths.py
scoring/      score_dm.py, score_dm_split.py, dm_common.py: DeepMind's scoring code, fetched and run unmodified
certifier/    certify3d.py, final_cert.py, pipeline.py, hazard.py, export_gpu.py, exact/ (exact re-check)
results/      RESULTS.md and the raw logs / JSON behind every number
```

Pose format: one row per pose, `(x, y, z, yaw, pitch, roll)`, with angles in degrees. The rotation is
`scipy.spatial.transform.Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True)`, applied to the body and then
translated by `(x, y, z)`. The body starts as the block [-4, 0] × [0, 1]². The first row is always
(-3, 0, 0, 0, 0, 0). The last row is (0, 6, 8, 90, 90, 90) for D2, D and B, and (0, 5, 9.2, 90, 180, 90) for the
double screw. The JSON files hold the same float64 values as the `.npy` files, written with `repr`, and read back bit
for bit (`python paths/check_paths.py` checks this).

## Setup

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt              # certifier, exact re-check, path checks (CPU)
pip install -r requirements-gpu.txt          # only for scoring/ (NVIDIA GPU, JAX with CUDA)
```

Run every command from this folder.

```
python paths/check_paths.py                  # JSON == npy, start/end rows, prints array SHA-256
(cd paths && shasum -a 256 -c SHA256SUMS)
```

## 1. Re-score with DeepMind's code

`scoring/score_dm.py` downloads `the_3d_moving_sofa.ipynb` from
`google-deepmind/alphaevolve_repository_of_problems` at commit `8f447457957deac61e28bf1676746f0753b3b2f8` and checks
its SHA-256 (`a5b03434…d73e`). It then executes the eight scoring cells unmodified and calls DeepMind's
`generate_sofa_and_evaluate_path` with tolerances 0.005 / 0.05°. The first and last rows of the pose file are dropped,
because DeepMind's code appends its own fixed start and end poses. The script also prints a float32 step-size guard
(the smallest true rotation per smoothed step). DeepMind's single-precision code returns NaN or 0 if a step is too
small, and the released paths were exported to avoid that (`certifier/export_gpu.py`).

```
python scoring/score_dm.py paths/D2.npy 0.001
#  expected: #### RESULT D2 grid 0.001 volume 1.8893629 survivors 1910309317

python scoring/score_dm.py paths/double_screw.npy 0.001 --end-pose-from-file
#  expected: #### RESULT double_screw grid 0.001 volume 1.9011554 survivors 1922601669  [END POSE CHANGED: ...]

python scoring/score_dm.py paths/D.npy 0.001        # expected 1.8870264, 1908096390
python scoring/score_dm.py paths/B.npy 0.001        # expected 1.8650145, 1884681159

python scoring/score_dm.py paths/D2.npy 0.001 --control-tang
#  also downloads Yuzhe Tang's published record path from his public gist (hash-checked; not stored here)
#  and scores it first. Expected: tang_w6_night3 1.8558404, 1876351816
```

* `--end-pose-from-file` is the only change to DeepMind's code that any result uses. It replaces the appended
  end pose (0, 6, 8, 90, 90, 90) with the file's last row. Without the flag the script refuses a file whose last row
  is not DeepMind's end pose.
* Quick check at grid 0.02 (seconds on a GPU): D2 gives 1.4920720 (235,332 points), D gives 1.4835200, the double
  screw with the flag gives 1.4893520, and Tang's path gives 1.4722400. At grid 0.002: D2 gives 1.8675882 (238,660,999).
* `--allow-cpu` runs on CPU JAX for coarse grids. CPU results are not the same as GPU results: D2 at grid 0.02 gave
  1.4933920 (235,550 points) on CPU with JAX 0.4.30, against 1.4920720 (235,332) on the GPU. Use a GPU for the
  reported numbers.
  The script writes `results_<name>_<grid>.json`.

**Several GPUs.** `score_dm_split.py` runs the same computation with the batch loop spread across machines. Batch `i`
holds the same 5×10^7 grid points as batch `i` of DeepMind's own loop, and grid 0.001 has 81 batches.

```
python scoring/score_dm_split.py work    d2 paths/D2.npy 0.001 0 41     # on machine 1
python scoring/score_dm_split.py work    d2 paths/D2.npy 0.001 41 81    # on machine 2
# copy every masks_d2/b*.npz into one masks_d2/ folder, then:
python scoring/score_dm_split.py combine d2 paths/D2.npy 0.001
#  expected: #### RESULT d2 grid 0.001 volume 1.8893629 survivors 1910309317 (split driver, 81 batches)
```

**Hardware and software used.**
* Our runs used a rented NVIDIA RTX 4090 (24 GB) with JAX 0.12.1. One path at grid 0.001 takes about 6 hours on one
  GPU (20,774–24,592 s in our runs).
* The published record was computed on an A100-class GPU with JAX 0.7.2. Our control run of that record path in our
  setup reproduced its published value exactly (1.8558404 at grid 0.001, and 1.8355162 at grid 0.002).

## 2. Re-run the independent certifier

```
python certifier/certify3d.py 6 paths/D2.npy
#  expected: paths/D2.npy  finest cell 0.000977  poses 7041  CERTIFIED VOLUME 1.906472  cells per level [...]

python certifier/final_cert.py 6 paths/D2.npy    # adds the connectivity check and negative controls
```

* Expected level-6 values: D2 1.906472, D 1.904259, B 1.880143, double screw 1.918910. The double screw is
  certified for its own end pose, and the certifier does not use DeepMind's appended end pose.
* Level 7 (`certify3d.py 7 …`) gives D2 1.911434, D 1.909281 and double screw 1.923975.
* A quick check at level 4 (under a minute) gives D2 1.876478.
* Time and memory: level 6 takes minutes on a multi-core CPU. `final_cert.py` at level 6 builds a 4096 × 1024 × 1024
  voxel grid (about 4.3 GB of RAM), and at level 7 that grows 8×. `certify3d.py` alone needs little memory.
* Numba caches compiled code in `certifier/__pycache__`. Set `NUMBA_CACHE_DIR` to put it elsewhere.

Other tools: `certifier/hazard.py <paths>` lists rotation steps too small for single precision.
`certifier/export_gpu.py` re-exports a pose list so that DeepMind's float32 code can score it, and a re-exported path is
a slightly different motion that must be re-certified.

## 3. Re-run the exact-arithmetic re-check

```
python -W ignore certifier/exact/export_cells.py D2 paths/D2.npy        # certified cubes -> exact_out/D2_cells.npz
python -W ignore certifier/exact/run_exact.py   D2 paths/D2.npy --all  # -> exact_out/D2_exact.json
#  expected: D2 poses 7041 cells per level [5409, 9331, 40278, 163046, 667179, 2689183, 10692647]
#            exact volume 2047058911/1073741824 = 1.906472175
#            ALL cubes (exact, skip variant): 14267073 / 14267073 pass, verified volume 1.906472175
```

* The re-check reports the negative controls too. Cubes shifted outward by 0.003 or scaled by 1.002 and 1.01 are
  rejected.
* `EXPORT_LEVELS=4` gives a quick run: for D2, 885,243 / 885,243 cubes pass, with verified volume 1.876477897.
* `EXACT_OUTDIR` changes the output folder.
* The run takes a few minutes per path at level 6.

## Credits

* **The problem and the scoring code.** Georgiev, Gómez-Serrano, Tao and Wagner, with Google DeepMind (AlphaEvolve).
  * Repository: `google-deepmind/alphaevolve_repository_of_problems`, code under Apache 2.0 and other material under
    CC-BY 4.0.
  * The scoring code is downloaded from that repository at run time and is not redistributed here.
* **The previous record and the staggered-twist architecture.** Yuzhe Tang, arXiv:2610.14051 and arXiv:2610.14050.
  * His pose data is not included here. `--control-tang` downloads it from his public gist when the user asks for it.
* **Path B** starts from the path published by AlphaEvolve in the repository above, and attribution is required.

## Caveats

* These are **lower bounds** only.
* They are **computer checks, not formal proofs**. The certifier and the exact re-check come with written soundness
  arguments (in the docstrings), but those arguments have not been formally verified.
* This work is **unreviewed**.
* Result 2 changes the end pose, so it does not count under DeepMind's fixed-end-pose formulation.
* DeepMind's score is a grid-based estimate defined by their code. Their code runs in single precision, and its
  results can depend on the GPU and the JAX version.
* The code and the write-up were built with the help of Claude, an AI assistant.

## Licence

Code (`scoring/`, `certifier/`, `paths/check_paths.py`): MIT, see `LICENSE`. Pose files and results (`paths/`, `results/`): CC BY 4.0, see `LICENSE-DATA`. Credits and attribution duties: `NOTICE.md`.
