"""Re-export a pose list so DeepMind's single-precision GPU code can score it.
Problem: that code returns NaN (volume 0) when a rotation sub-step is too small for float32. Our earlier exports had
(a) large Euler swings near pitch = 90 deg that are almost no real rotation, (b) near-pure translations.
Fix: (1) Euler angles chosen continuously BACKWARDS from the fixed end pose (90,90,90); within `p0` degrees of the
gimbal the ill-conditioned direction is dropped (the rotation is changed by at most about p0 degrees there, usually far less);
(2) poses densified so each step is short; (3) a tiny alternating pitch dither so every step has a real rotation.
The output is a slightly DIFFERENT motion, so it must be re-certified.
usage: export_gpu.py <in.npy> <out.npy> [p0_deg=0.25] [dither_deg=3e-4] [max_tr=0.01] [max_rot_deg=0.5]"""
import sys, numpy as np
from scipy.spatial.transform import Rotation as Rot

def E2R(e): return Rot.from_euler('zyx', e, degrees=True)

def solve(Rt, e0, s0):
    """Euler angles near e0 with R(e) ~ Rt; singular directions (sv < s0) are not used."""
    e = np.array(e0, float)
    for _ in range(60):
        r = (Rt * E2R(e).inv()).as_rotvec()                    # world-frame error
        if np.linalg.norm(r) < 1e-14: break
        J = np.zeros((3, 3)); d = 1e-6
        for k in range(3):
            ep = e.copy(); ep[k] += d
            J[:, k] = (E2R(ep) * E2R(e).inv()).as_rotvec() / np.radians(d)
        U, S, Vt = np.linalg.svd(J)
        keep = S > s0
        step = Vt.T[:, keep] @ ((U.T[keep] @ r) / S[keep])
        e += np.degrees(step)
        if np.abs(step).max() < 1e-15: break
    return e, (Rt * E2R(e).inv()).magnitude()

def export(P, p0=0.25, dither=3e-4, max_tr=0.01, max_rot=0.5):
    R = E2R(P[:, 3:]); t = P[:, :3].copy(); n = len(P)
    # densify on the geodesic
    Rs = [R[0]]; ts = [t[0]]
    for i in range(n - 1):
        rel = R[i + 1] * R[i].inv(); a = np.degrees(rel.magnitude()); d = np.linalg.norm(t[i + 1] - t[i])
        m = int(max(1, np.ceil(d / max_tr), np.ceil(a / max_rot)))
        rv = rel.as_rotvec()
        for j in range(1, m + 1):
            Rs.append(Rot.from_rotvec(rv * j / m) * R[i]); ts.append(t[i] + (t[i + 1] - t[i]) * j / m)
    N = len(Rs); E = np.zeros((N, 3)); res = np.zeros(N)
    E[-1] = P[-1, 3:]; s0 = np.sin(np.radians(p0))
    for i in range(N - 2, -1, -1):
        E[i], res[i] = solve(Rs[i], E[i + 1], s0)
    # bring the start to (0,0,0): by whole turns, and if needed by the equivalent branch (yaw+180, 180-pitch, roll+180)
    def wrap(E):
        sh = 360 * np.round(E[0] / 360); return E - sh, sh
    E1, shift = wrap(E)
    if np.abs(E1[0]).max() > 1.0:
        Ef = np.column_stack([E[:, 0] + 180, 180 - E[:, 1], E[:, 2] + 180])
        E1, shift = wrap(Ef)
    E = E1
    start_err = np.abs(E[0]).max()
    out = np.column_stack([np.array(ts), E])
    sgn = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    out[1:-1, 4] += dither * sgn[1:-1]
    out[0, 3:] = 0.0
    # the end pose keeps its rotation; its Euler triple may be an equivalent one (standard paths: unchanged)
    assert (E2R(out[-1, 3:]) * E2R(P[-1, 3:]).inv()).magnitude() < 1e-9 and start_err < 1e-6
    return out, res, start_err, shift

if __name__ == '__main__':
    P = np.load(sys.argv[1]); a = [float(x) for x in sys.argv[3:]]
    out, res, se, shift = export(P, *a)
    np.save(sys.argv[2], out)
    print('%s -> %s: poses %d -> %d  rotation changed by max %.2e deg (mean %.2e)  start mismatch %.2e deg  whole-turn shift %s  max euler step %.3f deg' % (
        sys.argv[1], sys.argv[2], len(P), len(out), np.degrees(res.max()), np.degrees(res.mean()), se, shift, np.abs(np.diff(out[:, 3:], axis=0)).max()))
