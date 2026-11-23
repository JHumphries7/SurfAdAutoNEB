"""Convex hull, stability-region vertices and the choice of bulk phases."""
import numpy as np
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from scipy.optimize import nnls

GASES = {"H", "N", "O", "F", "Cl"}     # elemental reference = isolated X2 molecule
TRIPLET = {"O"}                        # O2 ground state is a triplet
PIN = 1e-4                             # eV/atom below the hull when pinning a metastable host


def build_hull(entries, host):
    """Phase diagram with the host pinned onto the hull if it lies above it.
    Returns (pd, host_on_hull, e_hull, pd_unpinned)."""
    pd_raw = PhaseDiagram(entries)
    e_hull = pd_raw.get_e_above_hull(host, allow_negative=True)
    if e_hull <= 0:
        return pd_raw, host, e_hull, pd_raw
    pinned = PDEntry(host.composition, host.energy - (e_hull + PIN) * host.composition.num_atoms,
                     name=getattr(host, "name", "host"))
    others = [e for e in entries if e is not host]
    return PhaseDiagram([*others, pinned]), pinned, e_hull, pd_raw


def vertices(pd, host):
    """One dict per vertex of the host's stability region: neighbours, mu and delta-mu (eV)."""
    i_host = next(i for i, e in enumerate(pd.qhull_entries) if e is host)
    out = []
    for facet in pd.facets:
        if i_host not in facet:
            continue
        mu = pd._get_facet_chempots(facet)
        out.append({
            "neighbours": [pd.qhull_entries[i] for i in facet if i != i_host],
            "mu": {el.symbol: float(mu[el]) for el in mu},
            "dmu": {el.symbol: float(mu[el] - pd.el_refs[el].energy_per_atom) for el in mu},
        })
    return out


def reference_vertex(verts, host):
    """Rule: the vertex where the most electronegative element has the lowest delta-mu
    (the metal-rich side). Returns its index."""
    anion = max(host.composition.elements, key=lambda el: el.X).symbol
    return int(np.argmin([v["dmu"][anion] for v in verts]))


def _inside(comp, corners, elements):
    """True if comp is a mixture of the corner compositions."""
    vec = lambda c: [c.get_atomic_fraction(el) for el in elements]
    A = np.vstack([np.array([vec(c) for c in corners]).T, np.ones(len(corners))])
    return nnls(A, np.append(vec(comp), 1.0))[1] < 1e-6


def select_phases(entries, host, ehull=0.05, el_window=0.010, max_alt=2):
    """Bulk phases needed for the chemical potentials: [(entry, role)].
    Roles: host, vertex, elem_ref, elem_alt, near_hull."""
    pd, pinned, e_hull, pd_raw = build_hull(entries, host)
    verts = vertices(pd, pinned)
    picked = {}

    def add(entry, role):
        picked.setdefault(entry.entry_id, (entry, role))

    add(host, "host")
    neighbours = [e for v in verts for e in v["neighbours"]]
    for e in neighbours:
        add(e, "vertex")
    for el in pd_raw.elements:
        ref = pd_raw.el_refs[el]
        add(ref, "elem_ref")
        alts = [e for e in entries if e.composition.is_element and e.composition.elements[0] == el
                and e is not ref and e.energy_per_atom - ref.energy_per_atom <= el_window]
        for e in sorted(alts, key=lambda e: e.energy_per_atom)[:max_alt]:
            add(e, "elem_alt")
    corners = [host.composition] + [e.composition for e in neighbours]
    for e in entries:
        if len(e.composition.elements) > 1 and pd_raw.get_e_above_hull(e) <= ehull \
                and _inside(e.composition, corners, pd_raw.elements):
            add(e, "near_hull")
    return list(picked.values()), verts, e_hull, pd_raw
