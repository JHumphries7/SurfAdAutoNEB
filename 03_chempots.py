"""Step 3: chemical potentials from your own bulk energies. Writes chempots.json.

  python 03_chempots.py              (uses the metal-rich vertex)
  python 03_chempots.py --vertex 2   (choose another vertex from the printed list)
"""
import argparse
import os
import sys

from pymatgen.analysis.phase_diagram import PDEntry

from adatom import chempots, project, qe

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--vertex", type=int, help="number of the vertex to use as the reference")
a = ap.parse_args()

cfg = project.load_config()
phases = project.load_json("bulk/phases.json")

entries, host, missing = [], None, []
print(f"{'phase':22s}{'E (eV/atom)':>14s}{'P (kbar)':>10s}{'mag':>7s}")
for ph in phases:
    out = f"bulk/{ph['prefix']}_scf.pwo"
    r = qe.parse_pw(out) if os.path.exists(out) else {"ok": False, "problems": ["missing"]}
    if not r["ok"]:
        missing.append(f"{out}: {', '.join(r['problems'])}")
        continue
    entry = PDEntry(r["comp"], r["E"], name=ph["prefix"])
    entries.append(entry)
    if ph["role"] == "host":
        host = entry
    p = "" if ph["molecule"] or r["P_kbar"] is None else f"{r['P_kbar']:.1f}"
    print(f"{ph['prefix']:22s}{r['E'] / r['nat']:14.4f}{p:>10s}{r['mag'] if r['mag'] is not None else 0:7.2f}")
if missing:
    sys.exit("\nNot ready:\n  " + "\n  ".join(missing))

pd, pinned, e_hull, _ = chempots.build_hull(entries, host)
verts = chempots.vertices(pd, pinned)
ref = a.vertex - 1 if a.vertex else chempots.reference_vertex(verts, host)

print(f"\nhost is {1000 * e_hull:.1f} meV/atom {'above' if e_hull > 0 else 'below'} your own hull"
      + ("  -> pinned onto the hull" if e_hull > 0 else ""))
print("Vertices (mu in eV/atom, delta-mu relative to the elements):")
for i, v in enumerate(verts):
    mark = "  <- reference" if i == ref else ""
    print(f"  [{i + 1}] host + " + " + ".join(e.name for e in v["neighbours"]) + mark)
    for el in v["mu"]:
        print(f"        mu_{el:2s} = {v['mu'][el]:12.4f}    delta-mu = {v['dmu'][el]:8.4f}")

project.save_json("chempots.json", {
    "reference": ref, "e_hull_host_meV": 1000 * e_hull,
    "vertices": [{"neighbours": [e.name for e in v["neighbours"]], "mu": v["mu"], "dmu": v["dmu"]} for v in verts]})
print("\nwrote chempots.json")
print("next: python 04_choose_cases.py")
