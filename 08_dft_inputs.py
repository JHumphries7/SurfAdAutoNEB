"""Step 8: QE inputs for the minima and saddle points found by the MLIP.

  python 08_dft_inputs.py --all
For each hop h it writes, in dft/<case>/:
  <case>_min<h>_relax.pwi   relax of the minimum
  <case>_ts<h>_relax.pwi    saddle: relax with the adatom's x, y fixed
  <case>_ts<h>_scf.pwi      saddle: single point at the MLIP geometry (with --single-point)
"""
import argparse
import glob

from ase.io import read

from adatom import project, qe

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("cases", nargs="*", help="case ids or numbers")
ap.add_argument("--all", action="store_true")
ap.add_argument("--single-point", action="store_true", help="also write single-point inputs for the saddles")
a = ap.parse_args()

cfg = project.load_config()
meta = project.load_json("cases.json")["slabs"]

for c in project.pick_cases(a.cases, a.all):
    cid, m = c["id"], meta[c["slab"]]
    n_hops = len(glob.glob(f"neb/{cid}_hop*_TS.extxyz"))
    if not n_hops:
        raise SystemExit(f"no NEB results for {cid}: run 07_neb.py first")
    args = (cfg, m["kpts"], m["zcut"], m["emaxpos"])
    for h in range(n_hops):
        IS, TS = read(f"neb/{cid}_hop{h}_IS.extxyz"), read(f"neb/{cid}_hop{h}_TS.extxyz")
        qe.write_slab(f"dft/{cid}/{cid}_min{h}_relax.pwi", IS, *args)
        qe.write_slab(f"dft/{cid}/{cid}_ts{h}_relax.pwi", TS, *args, fix_xy_index=-1)
        if a.single_point:
            qe.write_slab(f"dft/{cid}/{cid}_ts{h}_scf.pwi", TS, *args, calculation="scf")
    print(f"{cid}: wrote inputs for {n_hops} hop(s) in dft/{cid}/")

print("\nRun every .pwi with pw.x (output: same name with .pwo), plus the clean-slab relaxations.")
print("next: python 09_results.py")
