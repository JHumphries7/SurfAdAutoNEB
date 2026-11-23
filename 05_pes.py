"""Step 5: energy map (PES) of the adatom over the surface with the MLIP. Run on a GPU node.

  python 05_pes.py --all
  python 05_pes.py 010_t1_Mo__Mo__a2
"""
import argparse
import os

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import BFGS

from adatom import pes_tools as pt
from adatom import project, slabs
from adatom.mlip import get_calc

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("cases", nargs="*", help="case ids or numbers")
ap.add_argument("--all", action="store_true")
ap.add_argument("--redo", action="store_true", help="recompute existing maps")
a = ap.parse_args()

cfg = project.load_config()
meta = project.load_json("cases.json")["slabs"]
backend, relax = cfg["mlip"]["backend"], cfg["mlip"]["relax_substrate"]
fmax, tol = cfg["mlip"]["fmax"], cfg["slab"]["plane_tol"]
os.makedirs("pes", exist_ok=True)

for tag, ads in sorted({(c["slab"], c["adatom"]) for c in project.pick_cases(a.cases, a.all)}):
    out = f"pes/{tag}__{ads}.npz"
    if os.path.exists(out) and not a.redo:
        print(f"{out} exists, skipping (use --redo to recompute)")
        continue
    m = meta[tag]
    prim = read(f"slabs/{tag}_1x1.extxyz")
    slab = prim.repeat((*m["repeat"], 1))

    # clean slab relaxed with the MLIP (bottom planes fixed)
    clean_file = f"pes/{tag}_clean.extxyz"
    if os.path.exists(clean_file) and not a.redo:
        slab = read(clean_file)
    else:
        if relax:
            slab.set_constraint(FixAtoms(mask=slab.positions[:, 2] < m["zcut"]))
            slab.calc = get_calc(backend)
            BFGS(slab, logfile=f"pes/{tag}_clean.log").run(fmax=fmax, steps=500)
            slab.set_constraint()
            slab.calc = None
        write(clean_file, slab)

    # grid: about grid_spacing apart; made finer if the surface symmetry needs it
    L = prim.cell.lengths()
    m1, m2 = (int(2 * np.ceil(x / cfg["pes"]["grid_spacing"] / 2)) for x in L[:2])
    big = max(m1, m2)
    tries = [(m1, m2, True), (6 * -(-big // 6),) * 2 + (True,), (12 * -(-big // 12),) * 2 + (True,), (m1, m2, False)]
    n_fix = m["n_fixed_planes"] if relax else len(slabs.planes(prim, tol))
    for g1, g2, sym in tries:
        try:
            res = pt.scan_pes(slab, prim, m["repeat"], ads, lambda: get_calc(backend), g1, g2,
                              n_fixed_layers=n_fix, relax_substrate=relax, fmax=fmax, steps=300,
                              optimizer=BFGS, layer_tol=tol, use_symmetry=sym, symprec=1e-2)
            break
        except ValueError:
            continue
    np.savez(out, E=res["E"], E_rel=res["E_rel"], z_ads=res["z_ads"], flags=res["flags"],
             a1=res["a1"], a2=res["a2"], origin=res["origin"])
    bad = int((res["flags"] > 0).sum())
    print(f"wrote {out}: {g1}x{g2} grid, corrugation {np.nanmax(res['E_rel']):.3f} eV, {bad} flagged points")

print("next: python 06_path.py")
