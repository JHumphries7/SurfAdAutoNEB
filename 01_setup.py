"""Step 1: start a project. Gets the conventional cell and writes config.yaml.

  python 01_setup.py --mp-id mp-2305 --api-key YOUR_KEY
  python 01_setup.py --cif MoC.cif --mp-id mp-2305
  python 01_setup.py --cif NbMoN.cif          (material not on Materials Project)
"""
import argparse
import copy
import os

from pymatgen.core import Structure

from adatom import mp, project

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mp-id", help="Materials Project id, e.g. mp-2305 (also used later for the convex hull)")
ap.add_argument("--cif", help="use this structure file instead of downloading one")
ap.add_argument("--keep-cell", action="store_true", help="with --cif: do not convert to the conventional cell")
ap.add_argument("--name", help="project name (default: the chemical formula)")
ap.add_argument("--api-key", help="Materials Project API key (or set MP_API_KEY)")
ap.add_argument("--pseudo-dir", default="./pseudos", help="folder with <Element>.upf files")
ap.add_argument("--ecutwfc", type=float, default=60, help="plane-wave cutoff, Ry")
ap.add_argument("--kspacing", type=float, default=0.25, help="k-point spacing, 1/Angstrom")
ap.add_argument("--nspin", type=int, default=2, choices=[1, 2])
ap.add_argument("--backend", default="esen_oam", help="MLIP: esen_oam, uma, mace, emt or test")
a = ap.parse_args()

if not a.mp_id and not a.cif:
    a.mp_id = input("Materials Project id (e.g. mp-2305): ").strip()

if a.cif:
    structure = Structure.from_file(a.cif)
    if not a.keep_cell:
        structure = mp.conventional(structure)
else:
    structure = mp.get_structure(a.mp_id, a.api_key)

if a.api_key:                              # remembered for step 2; listed in .gitignore
    open("mp_api_key.txt", "w").write(a.api_key)

formula = structure.composition.reduced_formula
name = a.name or formula
os.makedirs("structure", exist_ok=True)
path = f"structure/{name}.cif"
structure.to(filename=path)

cfg = copy.deepcopy(project.DEFAULTS)
cfg.update(name=name, mp_id=a.mp_id, structure=path)
cfg["qe"].update(pseudo_dir=os.path.abspath(a.pseudo_dir), ecutwfc=a.ecutwfc, kspacing=a.kspacing, nspin=a.nspin)
cfg["mlip"]["backend"] = a.backend
project.save_config(cfg)

abc, ang = structure.lattice.abc, structure.lattice.angles
print(f"{formula}: {len(structure)} atoms, space group {structure.get_space_group_info()[0]}")
print("  a, b, c = %.4f %.4f %.4f A   angles = %.1f %.1f %.1f" % (*abc, *ang))
print(f"wrote {path} and config.yaml  (edit config.yaml to change any setting)")
print("next: python 02_bulk_phases.py")
