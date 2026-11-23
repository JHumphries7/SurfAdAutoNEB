"""Step 9: collect the DFT energies into E_f, E_m and E_A = E_f + E_m. Writes results.csv.

  python 09_results.py
  python 09_results.py --vertex 2    (use another chemical-potential vertex)
"""
import argparse
import csv
import glob
import os

from adatom import project, qe

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--vertex", type=int, help="vertex number from 03_chempots.py")
ap.add_argument("--single-point", action="store_true", help="use the saddle single points (_scf) instead of _relax")
a = ap.parse_args()

chem = project.load_json("chempots.json")
vertex = chem["vertices"][a.vertex - 1 if a.vertex else chem["reference"]]
mu = vertex["mu"]
print("chemical potentials: host + " + " + ".join(vertex["neighbours"]),
      {k: round(v, 4) for k, v in mu.items()})


def energy(path):
    if not os.path.exists(path):
        return None, f"missing {path}"
    r = qe.parse_pw(path)
    return (r["E"], None) if r["ok"] else (None, f"{path}: {', '.join(r['problems'])}")


rows = []
for c in project.load_json("cases.json")["cases"]:
    cid, tag, ads = c["id"], c["slab"], c["adatom"]
    row = {"case": cid, "slab": tag, "adatom": ads, "direction": c["uvw"]}
    mlip = project.load_json(f"neb/{cid}.json", default={})
    row["E_m_mlip"] = mlip.get("E_m_mlip")
    kind = "scf" if a.single_point else "relax"
    n = len(glob.glob(f"dft/{cid}/{cid}_min*_relax.pwi"))
    files = [f"dft/{tag}/{tag}_clean_relax.pwo"]
    files += [f"dft/{cid}/{cid}_min{h}_relax.pwo" for h in range(n)]
    files += [f"dft/{cid}/{cid}_ts{h}_{kind}.pwo" for h in range(n)]
    values, problems = zip(*(energy(f) for f in files))
    problems = [p for p in problems if p]
    if not n:
        problems = ["no DFT inputs yet (run steps 5-8)"]
    if problems or ads not in mu:
        row["status"] = "; ".join(problems) or f"no chemical potential for {ads}"
    else:
        E_clean, E_min, E_ts = values[0], min(values[1:1 + n]), max(values[1 + n:])
        row.update(E_f=E_min - E_clean - mu[ads], E_m=E_ts - E_min, status="ok")
        row["E_A"] = row["E_f"] + row["E_m"]
    rows.append(row)

cols = ["case", "slab", "adatom", "direction", "E_f", "E_m", "E_A", "E_m_mlip", "status"]
with open("results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

fmt = lambda x: f"{x:8.3f}" if isinstance(x, float) else f"{'-':>8s}"
print(f"\n{'case':34s}{'E_f':>8s}{'E_m':>8s}{'E_A':>8s}{'E_m MLIP':>9s}  status")
for r in rows:
    print(f"{r['case']:34s}{fmt(r.get('E_f'))}{fmt(r.get('E_m'))}{fmt(r.get('E_A'))}{fmt(r.get('E_m_mlip')):>9s}  {r['status']}")
done = [r for r in rows if r["status"] == "ok"]
if done:
    best = min(done, key=lambda r: r["E_A"])
    print(f"\nlowest E_A: {best['E_A']:.3f} eV  ({best['case']})")
print("wrote results.csv (energies in eV)")
