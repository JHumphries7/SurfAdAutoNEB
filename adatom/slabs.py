"""Slab construction: atomic planes, terminations, flipping, fixed layers, in-plane directions."""
from math import gcd

import numpy as np
from ase import Atoms
from ase.build import surface
from ase.build.general_surface import ext_gcd
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list


def hkl_name(hkl):
    return "".join(f"m{-i}" if i < 0 else str(i) for i in hkl)


def planes(atoms, tol=0.3):
    """Atom indices grouped into atomic planes by z, bottom to top."""
    z = atoms.positions[:, 2]
    order = np.argsort(z)
    groups = [[order[0]]]
    for i in order[1:]:
        if z[i] - z[groups[-1][-1]] > tol:
            groups.append([i])
        else:
            groups[-1].append(i)
    return [np.array(sorted(g)) for g in groups]


def plane_formula(atoms, plane):
    return Atoms([atoms[i].symbol for i in plane]).get_chemical_formula()


def bond_cutoff(bulk):
    """1.15 x the shortest interatomic distance in the bulk."""
    d = neighbor_list("d", bulk, 6.0)
    return 1.15 * d.min()


def bulk_coordination(bulk, rcut):
    i = neighbor_list("i", bulk, rcut)
    cn = np.bincount(i, minlength=len(bulk))
    return {s: int(min(cn[k] for k, a in enumerate(bulk) if a.symbol == s)) for s in set(bulk.get_chemical_symbols())}


def flip(slab, vac_each):
    """Point inversion + in-plane wrap: puts the bottom face on top (any in-plane lattice)."""
    s = slab.copy()
    s.set_constraint()
    s.positions = -s.positions
    s.wrap(pbc=(True, True, False))
    s.positions[:, 2] += vac_each - s.positions[:, 2].min()
    return s


def terminations(bulk, hkl, n_planes=None, min_thickness=12.0, vacuum=15.0, tol=0.3):
    """Every distinct top termination of the (hkl) facet, including the opposite face.
    Returns a list of dicts with the 1x1 slab ('prim') and a description of its top surface."""
    vac_each = vacuum / 2
    one = planes(surface(bulk, hkl, 1, vacuum=None), tol)
    per_rep = len(one)
    probe = surface(bulk, hkl, 3, vacuum=None)
    P = planes(probe, tol)
    rep_height = probe.positions[P[per_rep], 2].mean() - probe.positions[P[0], 2].mean()
    if n_planes is None:               # whole repeat units, so the slab is stoichiometric
        zs = [probe.positions[p, 2].mean() for p in P[:per_rep + 1]]
        n_planes = per_rep * int(np.ceil((min_thickness + max(np.diff(zs))) / rep_height))
    stack = surface(bulk, hkl, int(np.ceil(n_planes / per_rep)) + 2, vacuum=None)
    P = planes(stack, tol)
    rcut = bond_cutoff(bulk)
    cn_bulk = bulk_coordination(bulk, rcut)

    found, seen = [], set()
    for start in range(per_rep):
        w = stack[np.concatenate(P[start:start + n_planes])]
        w.center(vacuum=vac_each, axis=2)
        w.pbc = True
        for side, prim in (("+", w), ("-", flip(w, vac_each))):
            q = planes(prim, tol)
            z = [prim.positions[p, 2].mean() for p in q]
            i = neighbor_list("i", prim, rcut)
            cn = np.bincount(i, minlength=len(prim))
            top = q[-1]
            sig = tuple((plane_formula(prim, q[-k]), round(z[-k] - z[-k - 1], 1)) for k in range(1, min(5, len(q))))
            sig += (tuple(sorted(cn[top].tolist())),)
            if sig in seen:
                continue
            seen.add(sig)
            kept = min(cn[k] / cn_bulk[prim[k].symbol] for k in top)
            found.append({
                "prim": prim, "side": side, "start": start, "n_planes": len(q),
                "top": plane_formula(prim, top), "below": plane_formula(prim, q[-2]),
                "gap": float(z[-1] - z[-2]),
                "bonds": ", ".join(f"{prim[k].symbol} {cn[k]}/{cn_bulk[prim[k].symbol]}" for k in top[:3]),
                "bonds_kept": float(kept),
                "thickness": float(z[-1] - z[0]),
            })
    return found


def fix_bottom(slab, tol=0.3, fraction=0.5):
    """Fix the lowest planes. Returns (slab, zcut, number of fixed planes)."""
    q = planes(slab, tol)
    n_fixed = max(1, int(len(q) * fraction))
    z = slab.positions[:, 2]
    zcut = 0.5 * (z[q[n_fixed - 1]].max() + z[q[n_fixed]].min())
    slab.set_constraint(FixAtoms(mask=z < zcut))
    return slab, float(zcut), n_fixed


def vacuum_middle(slab):
    """Fractional z of the middle of the vacuum (for the QE dipole correction)."""
    z = slab.positions[:, 2]
    c = slab.cell[2, 2]
    return float(round(((z.max() + 0.5 * (c - (z.max() - z.min()))) / c) % 1.0, 4))


def auto_repeat(prim, min_width):
    L = prim.cell.lengths()
    return [int(np.ceil(min_width / L[0] - 1e-6)), int(np.ceil(min_width / L[1] - 1e-6))]


def inplane_directions(bulk, hkl, prim):
    """Crystal directions [uvw] of the slab's in-plane vectors a1 (x axis) and a2."""
    h, k, l = hkl
    zeros = [i == 0 for i in hkl]
    if sum(zeros) == 2:
        c1, c2 = {0: ((0, 1, 0), (0, 0, 1)), 1: ((0, 0, 1), (1, 0, 0)), 2: ((1, 0, 0), (0, 1, 0))}[zeros.index(False)]
    else:                                  # same construction as ase.build.surface
        p, q = ext_gcd(k, l)
        a1, a2, a3 = bulk.cell
        k1 = np.dot(p * (k * a1 - h * a2) + q * (l * a1 - h * a3), l * a2 - k * a3)
        k2 = np.dot(l * (k * a1 - h * a2) - k * (l * a1 - h * a3), l * a2 - k * a3)
        if abs(k2) > 1e-10:
            i = -int(round(k1 / k2))
            p, q = p + i * l, q - i * k
        c1 = (p * k + q * l, -p * h, -q * h)
        c2 = tuple(int(x) for x in np.array((0, l, -k)) // abs(gcd(l, k)))
    out = []
    for name, c, vec in (("a1", c1, prim.cell[0]), ("a2", c2, prim.cell[1])):
        length = float(np.linalg.norm(vec))
        ok = abs(np.linalg.norm(np.dot(c, bulk.cell)) - length) < 1e-3
        uvw = "[" + " ".join(str(int(x)) for x in c) + "]" if ok else "[?]"
        out.append({"name": name, "uvw": uvw, "length": length})
    return out


def kpts_from_spacing(atoms, dk, periodic=(True, True, True)):
    """n_i = ceil(|b_i| / dk), with |b_i| including 2*pi (1/Angstrom)."""
    b = 2 * np.pi * np.linalg.norm(atoms.cell.reciprocal(), axis=1)
    return tuple(int(max(1, np.ceil(bi / dk - 1e-6))) if p else 1 for bi, p in zip(b, periodic))
