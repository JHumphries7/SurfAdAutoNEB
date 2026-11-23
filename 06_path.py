"""Step 6: find the migration path on the energy map and plot it.

  python 06_path.py --all
Writes pes/<case>_route.png (look at it), pes/<case>_route.json and one path file per hop.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from adatom import pes_tools as pt
from adatom import project, slabs

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("cases", nargs="*", help="case ids or numbers")
ap.add_argument("--all", action="store_true")
a = ap.parse_args()

cfg = project.load_config()
for c in project.pick_cases(a.cases, a.all):
    cid, tag, shift = c["id"], c["slab"], tuple(c["shift"])
    d = np.load(f"pes/{tag}__{c['adatom']}.npz")
    E, Z, F, a1, a2, org = d["E_rel"], d["z_ads"], d["flags"], d["a1"], d["a2"], d["origin"]
    prim = read(f"slabs/{tag}_1x1.extxyz")
    surf, surf_z = pt.PeriodicSurface(E, a1, a2, org), pt.PeriodicSurface(Z, a1, a2, org)

    # lowest-saddle route from the global minimum to its periodic image, then smoothed
    minima = pt.grid_minima(E, surf)
    start = np.unravel_index(np.argmin(E), E.shape)
    frac, _, E_grid = pt.minimax_path(E, a1, a2, start, shift=shift)
    xy = org + frac[:, :1] * a1[:2] + frac[:, 1:] * a2[:2]
    path, Ep, _ = pt.string_method(surf, xy, n_images=61, dt=0.02, n_iter=20000)
    i_min, i_max = pt.split_hops(Ep, prominence=2e-3)
    saddles = [pt.refine_saddle(surf, path[k]) for k in i_max]
    E0 = Ep[0]

    report = {
        "case": cid, "barrier_map_eV": float(Ep.max() - E0), "barrier_grid_eV": float(E_grid - E.min()),
        "n_hops": len(i_min) + 1,
        "minima": [{"xy": x.round(3).tolist(), "E": round(e, 4)} for x, e in minima],
        "intermediate_minima": [{"xy": path[k].round(3).tolist(), "E": round(float(Ep[k] - E0), 4)} for k in i_min],
        "saddles": [{"xy": s[0].round(3).tolist(), "E": round(s[1] - E0, 4),
                     "true_saddle": bool((s[2] < 0).sum() == 1)} for s in saddles],
        "flagged_points": int((F > 0).sum()),
    }
    project.save_json(f"pes/{cid}_route.json", report)

    for old in glob.glob(f"pes/{cid}_hop*.npy"):
        os.remove(old)
    cuts = [0, *i_min.tolist(), len(path) - 1]
    for h in range(len(cuts) - 1):
        np.save(f"pes/{cid}_hop{h}.npy", path[cuts[h]:cuts[h + 1] + 1])

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    top = prim[slabs.planes(prim, cfg["slab"]["plane_tol"])[-1]]
    pt.plot_pes(surf, top, ax=ax[0, 0], minima=minima, saddles=[s[0] for s in saddles], path_xy=path,
                vmax=min(float(E.max()), 1.5 * max(float(Ep.max() - E0), 0.05)))
    ax[0, 0].set_title(f"{c['adatom']} on {tag}: energy map and route along {c['uvw']}")
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    ax[0, 1].plot(arc, Ep - E0, "-o", ms=3)
    ax[0, 1].set_xlabel("distance along route (A)")
    ax[0, 1].set_ylabel("E - E(start) (eV)")
    ax[0, 1].set_title("energy along the route")
    pt.plot_pes(surf_z, top, ax=ax[1, 0], cmap="magma")
    fig.axes[-1].set_ylabel("adatom z (A)")
    ax[1, 0].set_title("relaxed adatom height")
    ax[1, 1].imshow(F.T, origin="lower", cmap="Reds", vmin=0, vmax=7)
    ax[1, 1].set_title("flagged grid points (0 = fine)")
    plt.tight_layout()
    plt.savefig(f"pes/{cid}_route.png", dpi=130)
    plt.close(fig)
    print(f"{cid}: barrier on the map {report['barrier_map_eV']:.3f} eV, {report['n_hops']} hop(s), "
          f"{report['flagged_points']} flagged points -> pes/{cid}_route.png")

print("next: python 07_neb.py   (on a GPU node)")
