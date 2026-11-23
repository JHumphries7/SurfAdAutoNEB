"""Optional check: neb.x input for a full DFT climbing-image NEB of one hop.

  python 10_nebx_input.py 010_t1_Mo__Mo__a2 --hop 0
Uses the DFT-relaxed minima as end points when their outputs exist, and the MLIP band in between.
Run with:  neb.x -inp <file>
"""
import argparse
import glob
import os

from ase.io import read

from adatom import pes_tools as pt
from adatom import project, qe

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("case")
ap.add_argument("--hop", type=int, default=0)
a = ap.parse_args()

cfg = project.load_config()
c = project.pick_cases([a.case], False)[0]
cid, m = c["id"], project.load_json("cases.json")["slabs"][c["slab"]]
images = read(f"neb/{cid}_hop{a.hop}_band.extxyz", index=":")
n_hops = len(glob.glob(f"neb/{cid}_hop*_TS.extxyz"))


def dft_min(h):
    out = f"dft/{cid}/{cid}_min{h}_relax.pwo"
    try:
        return qe.final_structure(out) if os.path.exists(out) else None
    except Exception:
        return None


first = dft_min(a.hop)
if first is not None:
    images[0].positions = first.positions
if a.hop + 1 < n_hops:
    last = dft_min(a.hop + 1)
elif dft_min(0) is not None:               # last hop ends at the periodic image of the first minimum
    ideal = read(f"slabs/{c['slab']}_1x1.extxyz").repeat((*m["repeat"], 1))
    shift = c["shift"][0] * ideal.cell[0] / m["repeat"][0] + c["shift"][1] * ideal.cell[1] / m["repeat"][1]
    last = pt.translate_state(dft_min(0), ideal, shift)
else:
    last = None
if last is not None:
    images[-1].positions = last.positions

path = f"dft/{cid}/{cid}_hop{a.hop}_neb.in"
qe.write_nebx(path, images, cfg, m["kpts"], m["zcut"], m["emaxpos"])
print(f"wrote {path}  ({len(images)} images; DFT end points: {first is not None}, {last is not None})")
