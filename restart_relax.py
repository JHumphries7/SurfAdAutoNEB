"""Continue a relax / vc-relax that stopped before converging (e.g. hit the BFGS step limit).

  python restart_relax.py bulk/MoC_host_vcrelax.pwo
Puts the last geometry (and cell) from the output into the matching .pwi, keeping every setting and
fixed-atom flag. The old files are kept as .pwi.1 / .pwo.1. Then run pw.x on the .pwi again.
"""
import os
import sys

from adatom.qe import final_structure

if len(sys.argv) != 2 or not sys.argv[1].endswith(".pwo"):
    sys.exit(__doc__)
pwo = sys.argv[1]
pwi = pwo[:-1] + "i"
try:
    last = final_structure(pwo)
except ValueError as err:
    sys.exit(f"{err}: just rerun {pwi} from scratch")
lines = open(pwi).read().splitlines()

for i, line in enumerate(lines):
    if line.startswith("CELL_PARAMETERS"):
        if "angstrom" not in line:
            sys.exit("expected CELL_PARAMETERS in angstrom")
        for k in range(3):
            lines[i + 1 + k] = " ".join(f"{x:.10f}" for x in last.cell[k])
    if line.startswith("ATOMIC_POSITIONS"):
        if "angstrom" not in line:
            sys.exit("expected ATOMIC_POSITIONS in angstrom")
        for k, pos in enumerate(last.positions):
            parts = lines[i + 1 + k].split()
            lines[i + 1 + k] = " ".join([parts[0]] + [f"{x:.10f}" for x in pos] + parts[4:])

for f in (pwi, pwo):
    n = 1
    while os.path.exists(f"{f}.{n}"):
        n += 1
    os.rename(f, f"{f}.{n}")
open(pwi, "w").write("\n".join(lines) + "\n")
print(f"updated {pwi} with the last geometry from {pwo}; run pw.x on it again")
