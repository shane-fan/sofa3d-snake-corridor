import numpy as np, sys, json, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the release/ folder
from certifier.certify3d import full_certificate
levels = int(sys.argv[1])
for name in sys.argv[2:]:
    P = np.array(json.load(open(name))['poses']) if name.endswith('.json') else np.load(name)
    out, grid = full_certificate(P, levels)
    print(name); [print('   %s: %s' % kv) for kv in out.items()]; sys.stdout.flush()
    del grid
