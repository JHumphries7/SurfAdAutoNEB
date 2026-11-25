"""Step 2: choose the bulk phases needed for the chemical potentials and write their QE inputs.

  python 02_bulk_phases.py
Writes vc-relax inputs (relax for gas molecules), or SCF inputs if relax_bulk is false in config.yaml.
If your material is not on Materials Project (no mp_id), every phase near the hull of its chemical
system is included, since its neighbours on the hull are not known until step 3.
Run every bulk/*.pwi with pw.x (output: same name with .pwo), then run 03_chempots.py.
"""
import argparse
import os
import sys

from ase.build import molecule
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from adatom import chempots, mp, project, qe, slabs

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--api-key")
ap.add_argument("--ehull", type=float, default=0.050, help="keep compounds this close to the hull, eV/atom")
ap.add_argument("--el-window", type=float, default=0.010, help="keep element polymorphs this close to the lowest")
ap.add_argument("--max-alt", type=int, default=2, help="extra polymorphs per element")
ap.add_argument("--box", type=float, default=20.0, help="box size for gas molecules, Angstrom")
ap.add_argument("--refresh", action="store_true", help="query Materials Project again")
a = ap.parse_args()

cfg = project.load_config()
host_struct = Structure.from_file(cfg.get("structure_unrelaxed") or cfg["structure"])   # always start from MP
elements = sorted(el.symbol for el in host_struct.composition.elements)

cache = "bulk/mp_entries.json"
os.makedirs("bulk", exist_ok=True)
if a.refresh or not os.path.exists(cache):
    dumpfn(mp.get_chemsys_entries(elements, a.api_key), cache + ".tmp")
    os.replace(cache + ".tmp", cache)      # only replace the cache once it is fully written
entries = mp.clean(loadfn(cache))

hosts = [e for e in entries if mp.mp_id_of(e) == cfg["mp_id"]] if cfg["mp_id"] else []
if cfg["mp_id"] and not hosts:
    same = [f"{mp.mp_id_of(e)} (entry {e.entry_id})" for e in entries
            if e.composition.reduced_formula == host_struct.composition.reduced_formula]
    sys.exit(f"{cfg['mp_id']} not found among the {len(entries)} GGA entries of {'-'.join(elements)}.\n"
             f"Entries with formula {host_struct.composition.reduced_formula}: {', '.join(same) or 'none'}\n"
             "If mp_entries.json came from an older version of this script, rerun with --refresh.")
if hosts:
    host = min(hosts, key=lambda e: e.energy_per_atom)
    phases, verts, e_hull, pd_raw = chempots.select_phases(entries, host, a.ehull, a.el_window, a.max_alt)
    print(f"{cfg['mp_id']} ({host.composition.reduced_formula}): {1000 * e_hull:.1f} meV/atom above the PBE hull"
          + ("  -> treated as on the hull (pinned)" if e_hull > 0 else ""))
    for v in verts:
        names = " + ".join(f"{e.composition.reduced_formula} ({mp.mp_id_of(e)})" for e in v["neighbours"])
        print("  vertex: host +", names, {k: round(x, 3) for k, x in v["dmu"].items()})
else:
    host = PDEntry(host_struct.composition, 0.0, name="host")        # energy comes from your own run
    phases, pd_raw = chempots.select_all_near_hull(entries, a.ehull, a.el_window, a.max_alt)
    phases = [(host, "host")] + phases
    print(f"{host_struct.composition.reduced_formula} is not taken from Materials Project: including every phase "
          f"within {1000 * a.ehull:.0f} meV/atom of the {'-'.join(elements)} hull.\n"
          "Its own hull energy and vertices are computed in step 3.")

dk = cfg["qe"]["kspacing"]
relax = cfg["relax_bulk"]
manifest = []
print(f"\n{'role':11s}{'phase':12s}{'mp id':14s}{'hull meV':>9s}  {'atoms':>5s}  k-points   input")
for entry, role in phases:
    formula = entry.composition.reduced_formula
    mpid = mp.mp_id_of(entry) if role != "host" or cfg["mp_id"] else "none"
    gas = entry.composition.is_element and entry.composition.elements[0].symbol in chempots.GASES
    magmom, kpts = 0.5, None
    if role == "host":
        atoms, prefix = AseAtomsAdaptor.get_atoms(host_struct), f"{formula}_host"
    elif gas:
        if role == "elem_alt":
            continue
        sym = entry.composition.elements[0].symbol
        atoms = molecule(f"{sym}2")
        atoms.set_cell([a.box] * 3)
        atoms.center()
        atoms.pbc = True
        formula, prefix, kpts = f"{sym}2", f"{sym}2_molecule", (1, 1, 1)
        magmom = 1.0 if sym in chempots.TRIPLET else 0.5
    else:
        atoms, prefix = AseAtomsAdaptor.get_atoms(mp.conventional(entry.structure)), f"{formula}_{mpid}"
    kpts = kpts or slabs.kpts_from_spacing(atoms, dk)
    calc = ("relax" if gas else "vc-relax") if relax else "scf"
    infile = f"bulk/{prefix}_{calc.replace('-', '')}.pwi"
    atoms.write(f"bulk/{prefix}.cif")
    qe.write_bulk(infile, atoms, cfg, kpts, calc, magmom)
    manifest.append({"prefix": prefix, "formula": formula, "mp_id": mpid, "role": role, "molecule": gas,
                     "input": infile})
    hull = f"{1000 * pd_raw.get_e_above_hull(entry):.1f}" if role != "host" or cfg["mp_id"] else "-"
    print(f"{role:11s}{formula:12s}{mpid:14s}{hull:>9s}  {len(atoms):5d}  "
          f"{'x'.join(map(str, kpts)):9s}  {infile}")

project.save_json("bulk/phases.json", manifest)
print("\nRun each input:  pw.x -in X.pwi > X.pwo   (inside bulk/)")
print("next: python 03_chempots.py")
