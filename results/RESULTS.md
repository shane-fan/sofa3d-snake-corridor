# Results and provenance

Every number below is backed by a raw log or JSON file in this folder, except where marked "log not included".
Paths A and C are not released (they are derived from third-party pose data whose licence is unclear); only their
final scores are listed, and lines about them were removed from the copied logs (see "Edits to copied logs").

## Path names

The logs use working names. Each one is tied to a released file by the SHA-256 of its float64 pose array
(`hashlib.sha256(P.tobytes())`, printed by `python paths/check_paths.py`), which the scoring logs print too.

| Released file | Names in the logs | Pose-array SHA-256 (prefix) | End pose |
|---|---|---|---|
| `paths/D2.npy` | `ours_D2`, `D2gpu`, `D2_gpu.npy` | `9daac154af27ad65` (printed in the D2 scoring logs) | (0, 6, 8, 90, 90, 90), DeepMind's fixed end pose |
| `paths/D.npy` | `ours_D`, `strictD`, `d_pol2_gpu.npy` | `c86e753cc722b366` (printed in the D scoring logs) | (0, 6, 8, 90, 90, 90) |
| `paths/B.npy` | `ours_B`, `aepolB`, `ae_polish_b_gpu_path.npy` | `9e2c9891ef6b227b` (the embedded copy the original scoring driver checked against) | (0, 6, 8, 90, 90, 90) |
| `paths/double_screw.npy` | `screw_A`, `screwAgpu`, `cand_A_gpu.npy` | `adc713c2b79f5d2a` (printed in `gpu-5/run_screw_A_0.001.log`) | (0, 5, 9.2, 90, 180, 90), **differs from DeepMind's** |
| (control, not included) | `tang_w6_night3` | n/a. Yuzhe Tang's published `w6_night3_path.json`, downloaded at run time and checked against the file's SHA-256 `4707af40…` | (0, 6, 8, 90, 90, 90) |

## 1. DeepMind's scoring code

Google DeepMind's own notebook code, executed unmodified (pinned commit, SHA-256 checked; see `scoring/`).
Tolerances: 0.005 translation, 0.05 degrees rotation. Hardware: NVIDIA RTX 4090 (24 GB), JAX 0.12.1, rented cloud
machines labelled gpu-1 … gpu-9 below. About 6 hours per path at grid 0.001 on one GPU (20,774–24,592 s).

The double screw row uses one disclosed change: the end pose that DeepMind's code appends is the file's own last
row (0, 5, 9.2, 90, 180, 90) instead of DeepMind's fixed (0, 6, 8, 90, 90, 90). It is not a score for DeepMind's
fixed-end-pose problem.

### Grid 0.001 (the resolution of the published records)

| Path | Volume | Surviving grid points | How run | Backing file(s) |
|---|---|---|---|---|
| Tang w6_night3 (control) | 1.8558404 | 1,876,351,816 | unsplit | `dm_scoring/gpu-4/RESULTS.txt`, `run_tang_w6_night3_0.001.log`, `results_tang_w6_night3_0.001.json` (volume 1.8558404089997955). The same value as his published one, so the setup reproduces the record. |
| B | 1.8650145 | 1,884,681,159 | unsplit | `dm_scoring/gpu-2/RESULTS.txt`, `run_ours_B_0.001.log`, `results_ours_B_0.001.json` |
| A (not released) | 1.8677106 | 1,888,257,315 | split, 81 batches | log not included |
| C (not released) | 1.8737911 | 1,894,408,860 | unsplit | log not included |
| D | 1.8870264 | 1,908,096,390 | unsplit **and** split (identical) | unsplit: `dm_scoring/gpu-6/RESULTS.txt`, `run_ours_D_0.001.log`, `results_ours_D_0.001.json`; split: `dm_scoring/split_combined/split_dr_result.txt` + batch logs `splitD_g*.log` on gpu-1, gpu-4, gpu-8, gpu-9 |
| **D2** | **1.8893629** | **1,910,309,317** | unsplit **and** split (identical) | unsplit: `dm_scoring/gpu-7/RESULTS.txt`, `run_ours_D2_0.001.log`, `results_ours_D2_0.001.json` (1.8893628569997918); split: `dm_scoring/split_combined/split_d2r_result.txt`, `dm_scoring/gpu-1/RESULTS.txt` + batch logs `split_g*.log` on gpu-1, gpu-8, gpu-9 |
| **double screw** (end pose changed) | **1.9011554** | **1,922,601,669** | unsplit | `dm_scoring/gpu-5/RESULTS.txt`, `run_screw_A_0.001.log` ("custom end pose … 0, 5, 9.2, 90, 180, 90"), `results_screw_A_0.001.json` |

Split runs: the 81 batches of 5×10^7 points were computed on several GPUs (`score_dm_split.py work`) and combined
(`combine`). For D and D2 the split and unsplit runs give the same volume and survivor count. A smaller
validation, D2 at grid 0.004, gave split = unsplit = 1.8235974 with 29,787,239 survivors (`dm_scoring/gpu-9/VALIDATION.txt`,
`val_split.log`, `val_unsplit.log`).

### Grid 0.002

| Path | Volume | Surviving grid points | Backing file(s) |
|---|---|---|---|
| Tang w6_night3 (control) | 1.8355162 | 234,544,396 | `dm_scoring/gpu-1/RESULTS.txt`, `results_tang_w6_night3_0.002.json`. Equals his published 0.002 value. |
| B | 1.8444762 | 235,456,095 | `dm_scoring/gpu-1/RESULTS.txt`, `results_ours_B_0.002.json` |
| A (not released) | 1.8470616 | 235,992,795 | log not included |
| C (not released) | 1.8521808 | 236,647,162 | log not included |
| D | 1.8650167 | 238,370,917 | `dm_scoring/gpu-6/RESULTS.txt`, `results_ours_D_0.002.json` |
| D2 | 1.8675882 | 238,660,999 | `dm_scoring/gpu-7/RESULTS.txt`, `results_ours_D2_0.002.json` |
| double screw (end pose changed) | 1.8784204 | 240,138,089 | `dm_scoring/gpu-2/RESULTS.txt` |

### Grid 0.02 (quick check only)

| Path | Volume | Surviving grid points | Backing file(s) |
|---|---|---|---|
| Tang w6_night3 (control) | 1.4722400 | 232,192 | every `RESULTS.txt` (first line), `results_tang_w6_night3_0.02.json` |
| D | 1.4835200 | 234,538 | `dm_scoring/gpu-6/results_ours_D_0.02.json` |
| D2 | 1.4920720 | 235,332 | `dm_scoring/gpu-7/results_ours_D2_0.02.json` |
| double screw (end pose changed) | 1.4893520 | 236,021 | `dm_scoring/gpu-2/RESULTS.txt` |

## 2. Independent certifier (`certifier/certify3d.py`)

This does not use DeepMind's point-and-cube rule. It certifies a union of octree cubes against a continuous motion,
with a screw-motion interpolation between poses and a rigorous chord-deviation bound (see the docstring). The root
cell is 1/16. Level 6 means the finest cube has side 1/1024 ≈ 0.00098, and level 7 means 1/2048 ≈ 0.00049.

| Path | Level 6 | Level 7 | Backing file(s) |
|---|---|---|---|
| Tang w6_night3 (control) | 1.870832 | n/a | `certifier/L6_tang_control_final_cert.log` (1.8708327682688832) |
| B | 1.880143 | n/a | `certifier/L6_B_certify3d.log` |
| A (not released) | 1.883058 | n/a | log not included |
| C (not released) | 1.889806 | n/a | log not included |
| D | 1.904259 | 1.909281 | L6: `certifier/L6_D_final_cert.log`; L7: `certifier/L7_D_and_double_screw_cert7.log` (run on gpu-4's CPU) |
| **D2** | **1.906472** | **1.911434** (reproduced on two machines, identical cell counts) | L6: `certifier/L6_D2_final_cert.log` (1.9064721753820777); L7: `certifier/L7_D2_cert_machine-a.log`, `certifier/L7_D2_cert_machine-b.log` |
| double screw (end pose changed) | 1.918910 | 1.923975 | L6: `certifier/L6_double_screw_final_cert.log`; L7: `certifier/L7_D_and_double_screw_cert7.log` |

The `final_cert` logs also report the negative controls, which pass. The connected component equals the full
certified set. Re-testing the certified shape scaled by 1.002 and 1.01 makes some cells fail, as expected. A Monte
Carlo check of 20,000 random points of the shape against the corridor, with the motion sampled 10× finer, finds 0
violations.

## 3. Exact-arithmetic re-check (`certifier/exact/`)

All level-6 certified cubes of each path are exported (`export_cells.py`) and re-verified with exact integer and
rational arithmetic and interval-arithmetic rotations (`run_exact.py --all`; the code shares nothing with
`certify3d.py`). It also checks the counts, disjointness and negative controls (cubes shifted outward by 0.003, or
scaled by 1.002 and 1.01, are rejected).

| Path | Cubes passing / total | Verified volume | Backing file(s) |
|---|---|---|---|
| Tang w6_night3 (control) | 13,448,144 / 13,448,144 | 1.870832768 | log not included (control path) |
| B | 13,444,527 / 13,444,527 | 1.880143894 | `exact/B/` |
| A (not released) | 13,225,436 / 13,225,436 | 1.883058969 | log not included |
| C (not released) | 14,006,477 / 14,006,477 | 1.889806435 | log not included |
| D | 14,460,212 / 14,460,212 | 1.904259124 | `exact/D/` |
| **D2** | **14,267,073 / 14,267,073** | **1.906472175** | `exact/D2/` (exact volume 2047058911/1073741824) |
| double screw (end pose changed) | 14,396,976 / 14,396,976 | 1.918910558 | `exact/double_screw/` |

In every case the exactly verified volume equals the level-6 certifier volume.

## Edits to copied logs

The logs were copied from the working directories with these edits only:

* Lines that named paths A or C were removed: their header lines (array shape and smoothed-step guard line) and
  their `#### RESULT` lines in `RESULTS.txt`. These are 6 lines per run log, 18 in `gpu-9/val_split.log` (three
  concatenated runs), and 1 line in each of `gpu-1/RESULTS.txt` and `gpu-6/RESULTS.txt`.
* The host directory prefix in front of the path file names was removed ("path file ours_D2.npy").
* Machine names were replaced by the neutral labels gpu-1 … gpu-9, which are the folder names here. In
  `split_combined/split_d2r_progress.log` the labels sA, sC and M1 became gpu-8, gpu-9 and gpu-1.
* Some files are excerpts, because the original file also covered unreleased paths:
  * `certifier/L6_B_certify3d.log` (one line)
  * `certifier/L6_tang_control_final_cert.log` (one block)
  * `exact/B/aepolB_export.log` (first 7 lines)
* Not copied: the logs for paths A and C, the logs for earlier unreleased path variants, and the batch mask files.
  The cube files (`*_cells.npz`, 3–8 MB each) can be regenerated with `export_cells.py`.

The scoring logs come from an earlier version of `scoring/score_dm.py` that also scored embedded copies of other
paths, so their headers list `tang_w6_night3` and `ours_B` before the path being scored. In those runs, files given
on the command line always had their own last row appended as the end pose. For D and D2 that row is exactly
DeepMind's (0, 6, 8, 90, 90, 90), so the appended poses are bit-identical to DeepMind's `add_start_and_end_poses`.
The released script uses DeepMind's function unless `--end-pose-from-file` is passed.
