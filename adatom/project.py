"""Project files (config.yaml, cases.json) and small command-line helpers."""
import json
import os
import sys

import yaml

DEFAULTS = {
    "name": None,
    "mp_id": None,
    "structure": None,
    "qe": {
        "pseudo_dir": "./pseudos",
        "ecutwfc": 60,
        "ecutrho": None,          # None = QE default (4 x ecutwfc)
        "smearing": "mv",
        "degauss": 0.01,
        "nspin": 2,
        "vdw_corr": "dft-d3",     # null = no dispersion correction
        "dftd3_version": 4,       # 4 = D3 with Becke-Johnson damping, 3 = zero damping
        "kspacing": 0.25,         # 1/Angstrom (2*pi included), used for bulk and slab k-meshes
        "conv_thr": 1e-7,
        "mixing_beta": 0.2,
        "forc_conv_thr": 3.9e-4,  # Ry/Bohr = 0.01 eV/Angstrom
    },
    "slab": {
        "min_thickness": 12.0,    # Angstrom; ignored if n_planes is set
        "n_planes": None,
        "vacuum": 15.0,           # total vacuum, Angstrom
        "min_width": 10.0,        # in-plane supercell is repeated to at least this length
        "repeat": None,           # e.g. [4, 4] to set the supercell by hand
        "plane_tol": 0.3,         # atoms within this z distance form one atomic plane
    },
    "mlip": {"backend": "esen_oam", "relax_substrate": True, "fmax": 0.02},
    "pes": {"grid_spacing": 0.18},
    "neb": {"n_images": 9, "fmax": 0.05},
}


def load_config():
    if not os.path.exists("config.yaml"):
        sys.exit("config.yaml not found. Run 01_setup.py in this folder first.")
    cfg = yaml.safe_load(open("config.yaml"))
    for section, values in DEFAULTS.items():      # fill in anything missing
        if isinstance(values, dict):
            cfg[section] = {**values, **(cfg.get(section) or {})}
        else:
            cfg.setdefault(section, values)
    return cfg


def save_config(cfg):
    yaml.safe_dump(cfg, open("config.yaml", "w"), sort_keys=False)


def load_json(path, default=None):
    if not os.path.exists(path):
        if default is None:
            sys.exit(f"{path} not found. Run the earlier steps first.")
        return default
    return json.load(open(path))


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(data, open(path, "w"), indent=1)


def parse_selection(text, n):
    """'all', '1,3', '2-4' or '1 3' -> list of zero-based indices."""
    text = text.strip().lower()
    if text in ("all", "a", "*"):
        return list(range(n))
    picked = []
    for part in text.replace(",", " ").split():
        lo, _, hi = part.partition("-")
        picked += range(int(lo) - 1, int(hi or lo))
    if not picked or min(picked) < 0 or max(picked) >= n:
        raise ValueError(f"choose numbers between 1 and {n}")
    return sorted(set(picked))


def choose(title, labels, preset=None):
    """Print a numbered list and return the chosen indices (from `preset` text or a prompt)."""
    print(f"\n{title}")
    for i, label in enumerate(labels, 1):
        print(f"  [{i}] {label}")
    if preset is not None:
        picked = parse_selection(preset, len(labels))
        print("  selected:", ", ".join(str(i + 1) for i in picked))
        return picked
    while True:
        try:
            return parse_selection(input("Select (e.g. 1,3 or 2-4 or all): "), len(labels))
        except (ValueError, IndexError) as err:
            print("  ", err)


def pick_cases(args_cases, use_all):
    """Cases named on the command line (id or number), or all, or chosen from a prompt."""
    cases = load_json("cases.json")["cases"]
    if not cases:
        sys.exit("No cases yet. Run 04_choose_cases.py first.")
    if use_all:
        return cases
    if not args_cases:
        idx = choose("Cases:", [c["id"] for c in cases])
        return [cases[i] for i in idx]
    by_id = {c["id"]: c for c in cases}
    out = []
    for a in args_cases:
        if a in by_id:
            out.append(by_id[a])
        elif a.isdigit() and 1 <= int(a) <= len(cases):
            out.append(cases[int(a) - 1])
        else:
            sys.exit(f"Unknown case '{a}'. Known cases:\n  " + "\n  ".join(by_id))
    return out
