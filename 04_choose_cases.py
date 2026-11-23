"""Step 4: choose facets, terminations, adatom species and migration directions.
Builds the slabs and writes the clean-slab QE relax inputs.

  python 04_choose_cases.py                      (interactive)
  python 04_choose_cases.py --list               (only show the options)
  python 04_choose_cases.py --facet 0 1 0 --terminations 1,3 --species Mo,C --directions all
"""
import argparse
import os

from ase.io import read, write
from pymatgen.core import Structure
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices

from adatom import project, qe, slabs

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--facet", type=int, nargs=3, action="append", metavar=("H", "K", "L"), help="repeat for more facets")
ap.add_argument("--max-index", type=int, default=1, help="largest Miller index in the facet list")
ap.add_argument("--terminations", help="e.g. 1,3 or all (numbers from the printed list)")
ap.add_argument("--species", help="e.g. Mo,C or all")
ap.add_argument("--directions", help="a1, a2 or all")
ap.add_argument("--planes", type=int, help="number of atomic planes in the slab")
ap.add_argument("--thickness", type=float, help="minimum slab thickness, Angstrom")
ap.add_argument("--repeat", type=int, nargs=2, metavar=("N1", "N2"), help="in-plane supercell")
ap.add_argument("--list", action="store_true", help="show the options and stop")
a = ap.parse_args()

cfg = project.load_config()
sl = cfg["slab"]
bulk = read(cfg["structure"])
elements = sorted(set(bulk.get_chemical_symbols()))
n_planes = a.planes or sl["n_planes"]
thickness = a.thickness or sl["min_thickness"]

# 1) facets
if a.facet:
    facets = [tuple(f) for f in a.facet]
else:
    options = get_symmetrically_distinct_miller_indices(Structure.from_file(cfg["structure"]), a.max_index)
    labels = ["(%d %d %d)" % tuple(f) for f in options]
    if a.list:
        facets = options
    else:
        facets = [options[i] for i in project.choose("Facets (symmetry-distinct, conventional cell):", labels)]

data = project.load_json("cases.json", default={"slabs": {}, "cases": []})
known = {c["id"] for c in data["cases"]}
species = None

for hkl in facets:
    name = slabs.hkl_name(hkl)
    terms = slabs.terminations(bulk, hkl, n_planes, thickness, sl["vacuum"], sl["plane_tol"])
    best = max(t["bonds_kept"] for t in terms)
    labels = [f"top: {t['top']:6s} on {t['below']:6s} gap {t['gap']:.2f} A | bonds kept: {t['bonds']:18s}"
              f"| {t['n_planes']} planes, {t['thickness']:.1f} A"
              + ("   <- fewest broken bonds" if t["bonds_kept"] == best else "") for t in terms]
    dirs = slabs.inplane_directions(bulk, hkl, terms[0]["prim"])
    dir_labels = [f"{d['name']} = {d['uvw']}, {d['length']:.3f} A" for d in dirs]

    if a.list:
        print(f"\nFacet ({hkl[0]} {hkl[1]} {hkl[2]})")
        for title, items in (("terminations", labels), ("adatom species", elements), ("directions", dir_labels)):
            print(f"  {title}:")
            for i, item in enumerate(items, 1):
                print(f"    [{i}] {item}")
        continue

    t_idx = project.choose(f"Terminations of ({hkl[0]} {hkl[1]} {hkl[2]}):", labels, a.terminations)
    if species is None:
        if a.species and a.species != "all" and not a.species[0].isdigit():
            species = [s.strip() for s in a.species.split(",")]
        else:
            species = [elements[i] for i in project.choose("Adatom species:", elements, a.species)]
    preset = a.directions.replace("a", "") if a.directions and a.directions != "all" else a.directions
    d_idx = project.choose("Migration directions (in-plane lattice vectors):", dir_labels, preset)

    for ti in t_idx:
        t = terms[ti]
        tag = f"{name}_t{ti + 1}_{t['top']}"
        prim = t["prim"]
        rep = a.repeat or sl["repeat"] or slabs.auto_repeat(prim, sl["min_width"])
        slab = prim.repeat((rep[0], rep[1], 1))
        slab, zcut, n_fixed = slabs.fix_bottom(slab, sl["plane_tol"])
        emaxpos = slabs.vacuum_middle(slab)
        kpts = slabs.kpts_from_spacing(slab, cfg["qe"]["kspacing"], periodic=(True, True, False))
        slab.set_constraint()
        os.makedirs("slabs", exist_ok=True)
        write(f"slabs/{tag}_1x1.extxyz", prim)
        write(f"slabs/{tag}.extxyz", slab)
        qe.write_slab(f"dft/{tag}/{tag}_clean_relax.pwi", slab, cfg, kpts, zcut, emaxpos)
        data["slabs"][tag] = {"hkl": list(hkl), "termination": labels[ti].split("   <-")[0], "top": t["top"],
                              "repeat": list(rep), "n_planes": t["n_planes"], "n_fixed_planes": n_fixed,
                              "natoms": len(slab), "zcut": zcut, "emaxpos": emaxpos, "kpts": list(kpts)}
        print(f"\nslab {tag}: {len(slab)} atoms, {rep[0]}x{rep[1]} cell, {t['n_planes']} planes ({n_fixed} fixed), "
              f"k = {'x'.join(map(str, kpts))}\n  wrote slabs/{tag}.extxyz and dft/{tag}/{tag}_clean_relax.pwi")
        for sp in species:
            for di in d_idx:
                cid = f"{tag}__{sp}__{dirs[di]['name']}"
                if cid not in known:
                    known.add(cid)
                    data["cases"].append({"id": cid, "slab": tag, "adatom": sp, "direction": dirs[di]["name"],
                                          "uvw": dirs[di]["uvw"], "shift": [1, 0] if di == 0 else [0, 1]})

if not a.list:
    project.save_json("cases.json", data)
    print("\nCases:")
    for i, c in enumerate(data["cases"], 1):
        print(f"  [{i}] {c['id']}   ({c['adatom']} adatom along {c['uvw']})")
    print("\nRun each dft/<slab>/<slab>_clean_relax.pwi with pw.x (output: same name with .pwo).")
    print("next: python 05_pes.py   (on a GPU node)")
