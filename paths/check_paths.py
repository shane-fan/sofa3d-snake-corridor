"""Check the released pose files.

Each path is stored twice: as a float64 NumPy array (<name>.npy) and as plain text (<name>.json).
Rows are (x, y, z, yaw, pitch, roll); angles in degrees, read as scipy Rotation.from_euler('zyx', ...).

usage (from the release/ folder):
    python paths/check_paths.py            # check JSON == npy bit for bit, start/end rows, print array SHA-256
    python paths/check_paths.py --write    # (re)write the .json files from the .npy files
"""
import sys, os, json, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['D2', 'D', 'B', 'double_screw']
START = [-3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
END = {'D2': [0.0, 6.0, 8.0, 90.0, 90.0, 90.0], 'D': [0.0, 6.0, 8.0, 90.0, 90.0, 90.0],
       'B': [0.0, 6.0, 8.0, 90.0, 90.0, 90.0], 'double_screw': [0.0, 5.0, 9.2, 90.0, 180.0, 90.0]}


def write_json(P, fname):
    # repr() of a Python float is the shortest string that reads back to the same binary64 value.
    rows = ',\n'.join('  [' + ', '.join(repr(float(v)) for v in row) + ']' for row in P)
    with open(fname, 'w') as f:
        f.write('{"columns": ["x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg"],\n'
                ' "euler_convention": "scipy Rotation.from_euler(\'zyx\', [yaw, pitch, roll], degrees=True)",\n'
                ' "poses": [\n' + rows + '\n]}\n')


def load(fname):
    if fname.endswith('.json'):
        d = json.load(open(fname))
        return np.array(d['poses'] if isinstance(d, dict) else d, dtype=np.float64)
    return np.load(fname).astype(np.float64)


def main():
    ok = True
    for n in NAMES:
        P = np.load(os.path.join(HERE, n + '.npy'))
        assert P.dtype == np.float64 and P.ndim == 2 and P.shape[1] == 6, n
        if '--write' in sys.argv:
            write_json(P, os.path.join(HERE, n + '.json'))
        Q = load(os.path.join(HERE, n + '.json'))
        same = Q.shape == P.shape and Q.tobytes() == P.tobytes()
        ends = P[0].tolist() == START and P[-1].tolist() == END[n]
        ok &= same and ends
        print('%-13s poses %5d  json==npy bit-exact: %s  start/end rows as expected: %s  array sha256 %s' % (
            n, len(P), same, ends, hashlib.sha256(P.tobytes()).hexdigest()))
    print('ALL OK' if ok else 'MISMATCH')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
