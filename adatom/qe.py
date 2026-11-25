"""Quantum ESPRESSO: write pw.x / neb.x inputs and read pw.x outputs."""
import io
import os
import re
from collections import Counter

from ase.constraints import FixAtoms, FixCartesian
from ase.io import read, write
from pymatgen.core import Composition

RY = 13.605693122994      # eV; the one conversion factor used everywhere


def _namelists(cfg, calculation, prefix, slab_dipole=None):
    q = cfg["qe"]
    control = {"calculation": calculation, "prefix": prefix, "pseudo_dir": q["pseudo_dir"],
               "outdir": "./tmp", "tprnfor": True}
    system = {"ecutwfc": q["ecutwfc"], "occupations": "smearing", "smearing": q["smearing"],
              "degauss": q["degauss"], "nspin": q["nspin"]}
    if q.get("ecutrho"):
        system["ecutrho"] = q["ecutrho"]
    if q.get("vdw_corr"):
        system["vdw_corr"] = q["vdw_corr"]
        if q.get("dftd3_version") and "d3" in q["vdw_corr"].lower():
            system["dftd3_version"] = q["dftd3_version"]
            system["dftd3_threebody"] = False  # two-body D3 only, to match the MLIP (three-body is too costly there)
    if slab_dipole is not None:            # dipole correction along z, sawtooth in the vacuum
        control.update(tefield=True, dipfield=True)
        system.update(edir=3, emaxpos=slab_dipole, eopreg=0.05, eamp=0.0)
    else:
        control["tstress"] = True
    data = {"control": control, "system": system,
            "electrons": {"conv_thr": q["conv_thr"], "mixing_mode": "local-TF", "mixing_beta": q["mixing_beta"]}}
    if calculation in ("relax", "vc-relax"):
        control["forc_conv_thr"] = q["forc_conv_thr"]
        control["nstep"] = q["nstep"]
        data["ions"] = {"ion_dynamics": "bfgs"}
    if calculation == "vc-relax":           # QE ends with a fresh SCF at the final cell
        data["cell"] = {"cell_dynamics": "bfgs", "cell_dofree": "all", "press_conv_thr": q["press_conv_thr"]}
    return data


def _write(path, atoms, cfg, data, kpts, magmom):
    if cfg["qe"]["nspin"] == 2:
        atoms.set_initial_magnetic_moments([magmom] * len(atoms))
    pseudos = {el: f"{el}.upf" for el in set(atoms.get_chemical_symbols())}
    buf = io.StringIO()
    write(buf, atoms, format="espresso-in", input_data=data, pseudopotentials=pseudos, kpts=kpts)
    text = buf.getvalue().replace("&FCP\n/\n", "").replace("&RISM\n/\n", "")   # not accepted by QE < 7.1
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write(text)
    return text


def write_bulk(path, atoms, cfg, kpts, calculation="vc-relax", magmom=0.5):
    """Bulk or molecule input: 'scf', 'relax' (positions only) or 'vc-relax' (positions and cell)."""
    prefix = os.path.basename(path).rsplit("_", 1)[0]
    _write(path, atoms.copy(), cfg, _namelists(cfg, calculation, prefix), kpts, magmom)


def write_slab(path, atoms, cfg, kpts, zcut, emaxpos, calculation="relax", fix_xy_index=None, magmom=0.5):
    """Slab input. zcut and emaxpos come from the CLEAN slab so every run of a slab matches."""
    atoms = atoms.copy()
    cons = [FixAtoms(mask=atoms.positions[:, 2] < zcut)]
    if fix_xy_index is not None:           # adatom: x, y fixed, z free
        cons.append(FixCartesian(fix_xy_index % len(atoms), mask=(True, True, False)))
    atoms.set_constraint(cons)
    prefix = os.path.basename(path).replace(".pwi", "")
    return _write(path, atoms, cfg, _namelists(cfg, calculation, prefix, slab_dipole=emaxpos), kpts, magmom)


def write_nebx(path, images, cfg, kpts, zcut, emaxpos, magmom=0.5):
    """neb.x input from a list of images (first, intermediates, last)."""
    def pw_text(atoms):
        tmp = path.rsplit(".", 1)[0] + ".pwi"
        text = write_slab(tmp, atoms, cfg, kpts, zcut, emaxpos, calculation="scf", magmom=magmom)
        os.remove(tmp)
        return text

    def positions(atoms):
        text = pw_text(atoms)
        return text[text.index("ATOMIC_POSITIONS"):].split("\n\n")[0].rstrip() + "\n"

    first = pw_text(images[0])
    head, tail = first.split("ATOMIC_POSITIONS")
    rest = tail[tail.index("\n\n") + 1:].strip() if "\n\n" in tail else ""
    n = cfg["neb"]["n_images"]
    out = ["BEGIN", "BEGIN_PATH_INPUT", "&PATH", "  string_method = 'neb'", "  restart_mode = 'from_scratch'",
           "  nstep_path = 200", f"  num_of_images = {n}", "  opt_scheme = 'broyden'", "  CI_scheme = 'auto'",
           "  path_thr = 0.05", "  ds = 1.0", "  k_max = 0.3", "  k_min = 0.2", "/", "END_PATH_INPUT",
           "BEGIN_ENGINE_INPUT", head.rstrip() + "\n", "BEGIN_POSITIONS", "FIRST_IMAGE", positions(images[0])]
    for im in images[1:-1]:
        out += ["INTERMEDIATE_IMAGE", positions(im)]
    out += ["LAST_IMAGE", positions(images[-1]), "END_POSITIONS", rest, "END_ENGINE_INPUT", "END"]
    open(path, "w").write("\n".join(out) + "\n")


def parse_pw(path):
    """Final energy (eV), composition, pressure and status of a pw.x output."""
    txt = open(path, errors="replace").read()
    tot = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", txt, re.M)
    if not tot:
        return {"file": path, "ok": False, "problems": ["no converged energy"]}
    nat = int(re.search(r"number of atoms/cell\s+=\s+(\d+)", txt).group(1))
    sites = re.findall(r"^\s+\d+\s+([A-Za-z]+\d*)\s+tau\(\s*\d+\)\s+=", txt, re.M)[:nat]
    press = re.findall(r"total\s+stress\s+\(Ry/bohr\*\*3\)\s+\(kbar\)\s+P=\s*(-?\d+\.\d+)", txt)
    mag = re.findall(r"total magnetization\s+=\s+(-?\d+\.\d+)", txt)
    problems = []
    if "JOB DONE" not in txt:
        problems.append("no JOB DONE")
    if "convergence NOT achieved" in txt:
        problems.append("an SCF did not converge")
    if "BFGS Geometry Optimization" in txt and "bfgs converged" not in txt:
        problems.append("relaxation not converged")
    return {"file": path, "ok": not problems, "problems": problems, "nat": nat,
            "comp": Composition(dict(Counter(re.sub(r"\d+$", "", s) for s in sites))),
            "E": float(tot[-1]) * RY,
            "P_kbar": float(press[-1]) if press else None,
            "mag": float(mag[-1]) if mag else None}


def final_structure(path):
    """Last geometry in a pw.x output (for restarts and neb.x end points)."""
    return read(path, index=-1, format="espresso-out")
