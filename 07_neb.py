"""Step 7: climbing-image NEB with the MLIP, one per hop of the route. Run on a GPU node.

  python 07_neb.py --all
Writes neb/<case>_hop<h>_{IS,TS,FS,band}.extxyz and neb/<case>.json.
"""
import argparse
import glob
import os

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import BFGS

from adatom import pes_tools as pt
from adatom import project
from adatom.mlip import d3_from_config, get_calc

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("cases", nargs="*", help="case ids or numbers")
ap.add_argument("--all", action="store_true")
a = ap.parse_args()

cfg = project.load_config()
meta = project.load_json("cases.json")["slabs"]
backend, relax = cfg["mlip"]["backend"], cfg["mlip"]["relax_substrate"]
d3 = d3_from_config(cfg)
os.makedirs("neb", exist_ok=True)

for c in project.pick_cases(a.cases, a.all):
    cid, tag, ads = c["id"], c["slab"], c["adatom"]
    m = meta[tag]
    d = np.load(f"pes/{tag}__{ads}.npz")
    a1, a2 = d["a1"], d["a2"]
    surf_z = pt.PeriodicSurface(d["z_ads"], a1, a2, d["origin"])
    ideal = read(f"slabs/{tag}_1x1.extxyz").repeat((*m["repeat"], 1))
    clean = read(f"pes/{tag}_clean.extxyz")
    paths = [np.load(p) for p in sorted(glob.glob(f"pes/{cid}_hop*.npy"))]
    if not paths:
        raise SystemExit(f"no route for {cid}: run 06_path.py first")

    def relaxed_state(xy):
        """Slab + adatom at xy, fully relaxed (adatom free)."""
        at = clean.copy() + Atoms(ads, positions=[[xy[0], xy[1], surf_z(xy)[0]]])
        fixed = at.positions[:, 2] < m["zcut"] if relax else np.arange(len(at)) < len(clean)
        at.set_constraint(FixAtoms(mask=fixed))
        at.calc = get_calc(backend, d3)
        BFGS(at, logfile=None).run(fmax=0.01, steps=1000)
        return at

    # end points: the minima along the route, then the periodic image of the first one
    states = [relaxed_state(p[0]) for p in paths]
    shift = c["shift"][0] * a1 + c["shift"][1] * a2
    states.append(pt.translate_state(states[0], ideal, shift))
    E_min = [s.get_potential_energy() for s in states[:-1]]

    hops = []
    for h, path in enumerate(paths):
        IS, FS = states[h], states[h + 1]
        path = path.copy()
        path[0], path[-1] = IS.positions[-1, :2], FS.positions[-1, :2]
        images = pt.neb_images_from_path(IS, FS, path, surf_z, n_images=cfg["neb"]["n_images"])
        calc = get_calc(backend, d3)
        for im in images:
            im.calc = calc
        neb = NEB(images, k=1.0, climb=False, method="improvedtangent", allow_shared_calculator=True)
        opt = BFGS(neb, logfile=f"neb/{cid}_hop{h}.log")
        opt.run(fmax=0.45, steps=200)                  # plain NEB first
        neb.climb = True
        converged = opt.run(fmax=cfg["neb"]["fmax"], steps=500)

        E = np.array([im.get_potential_energy() for im in images])
        ts = int(np.argmax(E))
        for name, im in (("IS", images[0]), ("TS", images[ts]), ("FS", images[-1])):
            write(f"neb/{cid}_hop{h}_{name}.extxyz", im)
        write(f"neb/{cid}_hop{h}_band.extxyz", images)
        hops.append({"hop": h, "converged": bool(converged), "saddle_is_interior": 0 < ts < len(images) - 1,
                     "E_saddle_minus_lowest_min": float(E.max() - min(E_min)),
                     "forward": float(E.max() - E[0]), "reverse": float(E.max() - E[-1])})
        print(f"{cid} hop {h}: forward {hops[-1]['forward']:.3f} eV, converged = {converged}")

    E_m = max(x["E_saddle_minus_lowest_min"] for x in hops)
    project.save_json(f"neb/{cid}.json", {"E_m_mlip": E_m, "hops": hops})
    print(f"{cid}: MLIP migration barrier E_m = {E_m:.3f} eV")

print("next: python 08_dft_inputs.py")
