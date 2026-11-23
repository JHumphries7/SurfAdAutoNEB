"""2D adatom PES scan, periodic interpolation, lowest-saddle path and NEB seeding (ASE).

The slab is prim.repeat((n1, n2, 1)) with the surface normal along +z; the adatom is the last atom.
Energies in eV, lengths in Angstrom."""
from __future__ import annotations

import heapq
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian, Hookean
from ase.data import covalent_radii, atomic_numbers
from ase.geometry import find_mic
from ase.optimize import BFGS, FIRE
from scipy import ndimage, optimize
from scipy.interpolate import make_interp_spline, NdBSpline


# --------------------------------------------------------------------------
# 1. Geometry helpers
# --------------------------------------------------------------------------
def layers_by_z(atoms: Atoms, tol: float = 0.3):
    """Atom indices grouped into planes by z, bottom to top."""
    z = atoms.positions[:, 2]
    order = np.argsort(z)
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if z[i] - z[cur[-1]] > tol:
            groups.append(np.array(sorted(cur)))
            cur = [i]
        else:
            cur.append(i)
    groups.append(np.array(sorted(cur)))
    return groups


def frac_inplane(xy, a1, a2):
    """Cartesian xy (...,2) -> fractional coords (...,2) w.r.t. a1,a2 (2D)."""
    B = np.array([a1[:2], a2[:2]]).T          # columns a1, a2
    return np.linalg.solve(B, np.asarray(xy, float).reshape(-1, 2).T).T


def cart_inplane(f, a1, a2, origin=(0.0, 0.0)):
    """fractional (...,2) -> Cartesian xy, always returns shape (N,2)."""
    f = np.atleast_2d(np.asarray(f, float))
    return origin + f[..., :1] * a1[:2] + f[..., 1:2] * a2[:2]


def pick_origin(slab: Atoms, top_idx):
    """xy of the top-layer atom closest to the supercell centre."""
    c = 0.5 * (slab.cell[0] + slab.cell[1])[:2]
    d = np.linalg.norm(slab.positions[top_idx, :2] - c, axis=1)
    return slab.positions[top_idx[np.argmin(d)], :2].copy()


def initial_height(slab: Atoms, xy, d0: float, cand_idx):
    """Start height at xy: lowest z keeping the adatom at least d0 from the surface atoms."""
    p = np.zeros((len(cand_idx), 3))
    p[:, :2] = slab.positions[cand_idx, :2] - np.asarray(xy)[None, :]
    p, _ = find_mic(p, slab.cell, pbc=[True, True, False])
    rho2 = (p[:, :2] ** 2).sum(1)
    zc = slab.positions[cand_idx, 2]
    ok = rho2 < d0 ** 2
    zmax = zc.max()
    if not ok.any():
        return zmax + 0.5 * d0
    return max((zc[ok] + np.sqrt(d0 ** 2 - rho2[ok])).max(), zmax + 0.3)


def bond_guess(sym_ads, sym_surf, scale=1.0):
    return scale * (covalent_radii[atomic_numbers[sym_ads]] +
                    covalent_radii[atomic_numbers[sym_surf]])


# --------------------------------------------------------------------------
# 2. Symmetry: irreducible grid points (spglib) and unfolding
# --------------------------------------------------------------------------
def surface_ops(prim: Atoms, symprec=1e-3):
    """In-plane symmetry operations of the 1x1 slab (fractional: f' = R f + t)."""
    import spglib
    cell = (prim.cell[:], prim.get_scaled_positions(), prim.numbers)
    ds = spglib.get_symmetry(cell, symprec=symprec)
    ops = []
    for R, t in zip(ds["rotations"], ds["translations"]):
        if R[2, 2] == 1 and R[2, 0] == 0 and R[2, 1] == 0 and \
           R[0, 2] == 0 and R[1, 2] == 0 and abs((t[2] + 0.5) % 1 - 0.5) < 1e-6:
            ops.append((R[:2, :2].copy(), t[:2].copy()))
    # remove duplicates (pure lattice translations already modded out)
    uniq = []
    for R, t in ops:
        if not any(np.array_equal(R, R2) and np.allclose((t - t2 + 0.5) % 1 - 0.5, 0, atol=1e-6)
                   for R2, t2 in uniq):
            uniq.append((R, t))
    return uniq


def irreducible_grid(m1, m2, ops, f_origin):
    """Map every grid node to a symmetry representative. Raises ValueError if the grid
    does not fit the symmetry (change m1/m2)."""
    rep = -np.ones((m1, m2), int)
    nodes = []
    for i in range(m1):
        for j in range(m2):
            if rep[i, j] >= 0:
                continue
            f = f_origin + np.array([i / m1, j / m2])
            k = len(nodes)
            nodes.append((i, j))
            for R, t in ops:
                g = R @ f + t - f_origin
                gi = g * np.array([m1, m2])
                gr = np.rint(gi)
                if np.abs(gi - gr).max() > 1e-4:
                    raise ValueError("grid not commensurate with symmetry op; "
                                     "change m1/m2 (or set use_symmetry=False)")
                ii, jj = int(gr[0]) % m1, int(gr[1]) % m2
                if rep[ii, jj] < 0:
                    rep[ii, jj] = k
    return np.array(nodes), rep          # rep[i,j] -> index into nodes


# --------------------------------------------------------------------------
# 3. The scan
# --------------------------------------------------------------------------
def scan_pes(slab: Atoms, prim: Atoms, n_rep, ads_symbol: str, calc_factory,
             m1: int, m2: int, n_fixed_layers: int, *, relax_substrate=True,
             d0=None, fmax=0.02, steps=300, optimizer=BFGS, layer_tol=0.3,
             use_symmetry=True, symprec=1e-3, warm_start=True,
             z_floor_offset=0.3, wall_k=10.0, traj_prefix=None, verbose=True):
    """Constrained PES scan: adatom x, y fixed and z free; substrate relaxed except the
    bottom n_fixed_layers planes (or fully frozen if relax_substrate=False).
    Flags: 1 = adatom hit the floor, 2 = not converged, 4 = surface atom moved > 0.5 A."""
    n1, n2 = n_rep
    a1, a2 = prim.cell[0].copy(), prim.cell[1].copy()
    layers = layers_by_z(slab, layer_tol)
    top = layers[-1]
    cand = np.concatenate(layers[-3:])
    fixed = np.concatenate(layers[:n_fixed_layers]) if n_fixed_layers else np.array([], int)
    assert abs(a1[2]) < 1e-8 and abs(a2[2]) < 1e-8 and \
        np.allclose(prim.cell[2][:2], 0), "need a1,a2 in-plane and a3 along z"
    assert np.allclose(slab.cell[0], n1 * a1) and np.allclose(slab.cell[1], n2 * a2), \
        "slab must be prim.repeat((n1, n2, 1))"
    origin = pick_origin(slab, top)
    # prim and slab share the same cell origin (slab = prim.repeat), so the
    # prim-frame fractional coordinate of any slab xy is frac_inplane(xy) mod 1.
    f_origin = frac_inplane(origin, a1, a2)[0] % 1.0
    if d0 is None:
        d0 = bond_guess(ads_symbol, slab[top[0]].symbol)

    if use_symmetry:
        ops = surface_ops(prim, symprec)
        nodes, rep = irreducible_grid(m1, m2, ops, f_origin)
    else:
        ops = []
        nodes = np.array([(i, j) for i in range(m1) for j in range(m2)])
        rep = np.arange(m1 * m2).reshape(m1, m2)
    if verbose:
        print(f"[scan] {len(ops)} in-plane ops, {len(nodes)} irreducible of {m1*m2} points")

    # serpentine order over irreducible nodes -> small moves between warm starts
    order = sorted(range(len(nodes)),
                   key=lambda k: (nodes[k][0], nodes[k][1] if nodes[k][0] % 2 == 0 else -nodes[k][1]))

    base = slab.copy()
    calc = calc_factory()
    z_top0 = slab.positions[top, 2].max()
    E_nodes = np.full(len(nodes), np.nan)
    z_nodes = np.full(len(nodes), np.nan)
    flag_nodes = np.zeros(len(nodes), int)       # bit0 wall active, bit1 not converged, bit2 surface atom moved >0.5 A
    prev = None
    for cnt, k in enumerate(order):
        i, j = nodes[k]
        xy = cart_inplane(np.array([i / m1, j / m2]), a1, a2, origin)[0]
        if warm_start and prev is not None:
            atoms = prev.copy()
            atoms.positions[-1, :2] = xy
            atoms.positions[-1, 2] = max(atoms.positions[-1, 2],
                                         initial_height(atoms[:-1], xy, 0.85 * d0, cand))
        else:
            atoms = base.copy()
            z0 = initial_height(base, xy, d0, cand)
            atoms += Atoms(ads_symbol, positions=[[xy[0], xy[1], z0]])
        ia = len(atoms) - 1
        cons = []
        if relax_substrate:
            if len(fixed):
                cons.append(FixAtoms(indices=fixed))
        else:
            cons.append(FixAtoms(indices=np.arange(ia)))
        cons.append(FixCartesian(ia, mask=(True, True, False)))
        z_floor = z_top0 + z_floor_offset          # adatom may not sink below top plane + offset
        cons.append(Hookean(a1=ia, a2=(0.0, 0.0, -1.0, z_floor), k=wall_k))
        atoms.set_constraint(cons)
        atoms.calc = calc
        opt = optimizer(atoms, logfile=None,
                        trajectory=None if traj_prefix is None else f"{traj_prefix}_{i:03d}_{j:03d}.traj")
        conv = opt.run(fmax=fmax, steps=steps)
        # energy WITHOUT the Hookean penalty:
        E = atoms.get_potential_energy(apply_constraint=False)
        fl = 0
        if atoms.positions[ia, 2] < z_floor + 0.05:
            fl |= 1
        if not conv:
            fl |= 2
        if relax_substrate and np.abs(atoms.positions[top] - base.positions[top]).max() > 0.5:
            fl |= 4
        E_nodes[k], z_nodes[k], flag_nodes[k] = E, atoms.positions[ia, 2], fl
        prev = atoms.copy()
        prev.calc = None
        if verbose and (cnt % max(1, len(nodes) // 10) == 0):
            print(f"[scan] {cnt+1}/{len(nodes)} node ({i},{j}) E={E:.4f} z={atoms.positions[ia,2]:.3f} flag={fl}")
    E = E_nodes[rep]
    Z = z_nodes[rep]
    F = flag_nodes[rep]
    return dict(E=E, E_rel=E - np.nanmin(E), z_ads=Z, flags=F, a1=a1, a2=a2,
                origin=origin, m1=m1, m2=m2, nodes=nodes, rep=rep, d0=d0)


# --------------------------------------------------------------------------
# 4. Periodic interpolation of a gridded quantity
# --------------------------------------------------------------------------
class PeriodicSurface:
    """Exactly periodic cubic-spline interpolant of grid data V[i, j]; evaluate with Cartesian xy."""

    def __init__(self, V, a1, a2, origin):
        V = np.asarray(V, float)
        m1, m2 = V.shape
        self.m1, self.m2 = m1, m2
        self.a1, self.a2, self.origin = np.asarray(a1)[:2], np.asarray(a2)[:2], np.asarray(origin, float)[:2]
        u = np.arange(m1 + 1) / m1
        v = np.arange(m2 + 1) / m2
        s1 = make_interp_spline(u, np.concatenate([V, V[:1]], 0), k=3, bc_type="periodic", axis=0)
        c1 = s1.c
        s2 = make_interp_spline(v, np.concatenate([c1, c1[:, :1]], 1), k=3, bc_type="periodic", axis=1)
        self.nd = NdBSpline((s1.t, s2.t), np.moveaxis(s2.c, 0, 1), 3)
        self.B = np.array([self.a1, self.a2]).T
        self.Binv = np.linalg.inv(self.B)

    def _f(self, xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        f = (self.Binv @ (xy - self.origin).T).T
        return f % 1.0

    def __call__(self, xy):
        return self.nd(self._f(xy))

    def grad(self, xy):
        f = self._f(xy)
        g = np.stack([self.nd(f, nu=(1, 0)), self.nd(f, nu=(0, 1))], axis=1)   # dV/df
        return g @ self.Binv                                                   # dV/dr

    def hessian(self, xy):
        f = self._f(xy)
        Hf = np.array([[self.nd(f, nu=(2, 0))[0], self.nd(f, nu=(1, 1))[0]],
                       [self.nd(f, nu=(1, 1))[0], self.nd(f, nu=(0, 2))[0]]])
        return self.Binv.T @ Hf @ self.Binv


# --------------------------------------------------------------------------
# 5. Minima, minimax (bottleneck) path on the periodic grid, string method
# --------------------------------------------------------------------------
def grid_minima(E, surf: PeriodicSurface, dedupe=0.2):
    """Local minima of the grid, refined on the spline. Returns [(xy, E)] sorted by energy."""
    mask = ndimage.minimum_filter(E, size=3, mode="wrap") == E
    out = []
    for i, j in zip(*np.nonzero(mask)):
        xy0 = cart_inplane(np.array([i / surf.m1, j / surf.m2]), surf.a1, surf.a2, surf.origin)[0]
        r = optimize.minimize(lambda p: surf(p)[0], xy0, jac=lambda p: surf.grad(p)[0], method="BFGS")
        xy = r.x
        f = surf._f(xy)[0]
        xy = cart_inplane(f, surf.a1, surf.a2, surf.origin)[0]   # wrap into home cell
        if all(np.linalg.norm(find_mic(np.r_[xy - q, 0], np.c_[np.r_[surf.a1, 0], np.r_[surf.a2, 0], [0, 0, 1]].T,
                                       pbc=[True, True, False])[0]) > dedupe for q, _ in out):
            out.append((xy, float(r.fun)))
    return sorted(out, key=lambda t: t[1])


def _tiled_graph(E, a1, a2, T1, T2):
    m1, m2 = E.shape
    ET = np.tile(E, (T1, T2))
    N1, N2 = ET.shape
    nbrs = [(1, 0), (0, 1), (1, 1), (1, -1)]
    edges = []
    for di, dj in nbrs:
        L = np.linalg.norm(di / m1 * a1[:2] + dj / m2 * a2[:2])
        I, J = np.meshgrid(np.arange(N1), np.arange(N2), indexing="ij")
        I2, J2 = I + di, J + dj
        ok = (I2 >= 0) & (I2 < N1) & (J2 >= 0) & (J2 < N2)
        edges.append((I[ok] * N2 + J[ok], I2[ok] * N2 + J2[ok], np.full(ok.sum(), L)))
    s = np.concatenate([e[0] for e in edges]); t = np.concatenate([e[1] for e in edges])
    L = np.concatenate([e[2] for e in edges])
    return ET, s, t, L


def minimax_path(E, a1, a2, ij_start, shift=(0, 1), tiles=(3, 3)):
    """Lowest-saddle path on the periodic grid from node ij_start to its periodic image
    shifted by `shift` surface lattice vectors. Returns (fractional path, E along it, E_saddle)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    m1, m2 = E.shape
    T1, T2 = tiles
    ET, s, t, L = _tiled_graph(E, a1, a2, T1, T2)
    N1, N2 = ET.shape
    c1, c2 = T1 // 2, T2 // 2
    i0, j0 = ij_start
    src = (c1 * m1 + i0) * N2 + (c2 * m2 + j0)
    if isinstance(shift, tuple) and len(shift) == 3 and shift[0] == "node":
        (i1, j1), (s1, s2) = shift[1], shift[2]
    else:
        (i1, j1), (s1, s2) = (i0, j0), shift
    if not (0 <= c1 + s1 < T1 and 0 <= c2 + s2 < T2):
        raise ValueError("increase tiles")
    dst = ((c1 + s1) * m1 + i1) * N2 + ((c2 + s2) * m2 + j1)
    flat = ET.ravel()
    # --- union-find flooding
    parent = np.arange(flat.size)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    adj = [[] for _ in range(flat.size)]
    for a, b in zip(s, t):
        adj[a].append(b); adj[b].append(a)
    active = np.zeros(flat.size, bool)
    Es = None
    for n in np.argsort(flat, kind="stable"):
        active[n] = True
        for b in adj[n]:
            if active[b]:
                ra, rb = find(n), find(b)
                if ra != rb:
                    parent[ra] = rb
        if active[src] and active[dst] and find(src) == find(dst):
            Es = flat[n]
            break
    # --- restricted Dijkstra
    emin = flat.min()
    eps = 1e-3 + 1e-2 * (flat.max() - emin)
    keep = (flat[s] <= Es + 1e-12) & (flat[t] <= Es + 1e-12)
    w = L[keep] * (0.5 * (flat[s][keep] + flat[t][keep]) - emin + eps)
    G = coo_matrix((w, (s[keep], t[keep])), shape=(flat.size, flat.size)).tocsr()
    G = G + G.T
    dist, pred = dijkstra(G, directed=False, indices=src, return_predecessors=True)
    if not np.isfinite(dist[dst]):
        raise RuntimeError("no path under threshold (should not happen)")
    path = [dst]
    while path[-1] != src:
        path.append(pred[path[-1]])
    path = path[::-1]
    I, J = np.divmod(np.array(path), N2)
    f = np.stack([(I - c1 * m1) / m1, (J - c2 * m2) / m2], axis=1)
    return f, flat[path], Es


def string_method(surf: PeriodicSurface, xy_init, n_images=41, dt=0.01, n_iter=5000,
                  tol=1e-6, fix_ends=False):
    """Zero-temperature string method on the interpolated surface (relaxes a path to the MEP)."""
    def resample(P, n):
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        s = np.linspace(0, d[-1], n)
        return np.stack([np.interp(s, d, P[:, k]) for k in range(2)], axis=1)
    P = resample(np.asarray(xy_init, float), n_images)
    for it in range(n_iter):
        g = surf.grad(P)
        if fix_ends:
            g[0] = g[-1] = 0
        Pn = resample(P - dt * g, n_images)
        if np.abs(Pn - P).max() < tol:
            P = Pn
            break
        P = Pn
    return P, surf(P), it


def refine_saddle(surf: PeriodicSurface, xy_guess):
    """Stationary point near xy_guess: returns (xy, E, Hessian eigenvalues)."""
    r = optimize.root(lambda p: surf.grad(p)[0], xy_guess, method="hybr")
    xy = r.x
    ev = np.linalg.eigvalsh(surf.hessian(xy))
    return xy, float(surf(xy)[0]), ev


def split_hops(E_path, prominence=1e-3):
    """Indices of interior local minima/maxima along a 1D energy profile."""
    from scipy.signal import find_peaks
    mx, _ = find_peaks(E_path, prominence=prominence)
    mn, _ = find_peaks(-E_path, prominence=prominence)
    return mn, mx


# --------------------------------------------------------------------------
# 6. NEB seeding
# --------------------------------------------------------------------------
def translate_state(state: Atoms, ref_slab: Atoms, shift_cart, ads_index=-1, tol=0.3):
    """Copy of a relaxed slab+adatom state shifted by a surface lattice vector, with
    substrate atoms re-indexed so it can be used as an NEB end point."""
    out = state.copy()
    n = len(ref_slab)
    P0 = ref_slab.positions
    shift_cart = np.asarray(shift_cart, float)
    # map: for each target site i, source j with P0[j] + shift == P0[i] (mod cell)
    src = np.empty(n, int)
    for i in range(n):
        d = P0 + shift_cart - P0[i]
        d, dl = find_mic(d, ref_slab.cell, pbc=[True, True, False])
        cand = np.nonzero((dl < tol) & (ref_slab.numbers == ref_slab.numbers[i]))[0]
        if len(cand) != 1:
            raise ValueError("shift is not a lattice translation of the slab")
        src[i] = cand[0]
    Ps = state.positions[:n]
    newP = Ps[src] + shift_cart
    dv, _ = find_mic(newP - Ps, state.cell, pbc=[True, True, False])
    out.positions[:n] = Ps + dv
    ia = ads_index % len(state)
    out.positions[ia] = state.positions[ia] + shift_cart
    return out


def neb_images_from_path(IS: Atoms, FS: Atoms, path_xy, surf_z: PeriodicSurface,
                         n_images: int, ads_index=-1, dz=0.0):
    """NEB images: adatom follows the PES path, substrate interpolated between IS and FS."""
    P = np.asarray(path_xy, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.linspace(0, d[-1], n_images)
    xy = np.stack([np.interp(s, d, P[:, k]) for k in range(2)], axis=1)
    ia = ads_index % len(IS)
    imgs = [IS.copy()]
    for k in range(1, n_images - 1):
        lam = k / (n_images - 1)
        im = IS.copy()
        im.positions = (1 - lam) * IS.positions + lam * FS.positions
        im.positions[ia, :2] = xy[k]
        im.positions[ia, 2] = surf_z(xy[k])[0] + dz
        imgs.append(im)
    imgs.append(FS.copy())
    return imgs


# --------------------------------------------------------------------------
# 7. Plotting
# --------------------------------------------------------------------------
def plot_pes(surf: PeriodicSurface, prim_top: Atoms, ax=None, window=((-0.5, 1.5), (-0.5, 1.5)),
             n_fine=200, levels=30, minima=None, saddles=None, path_xy=None,
             cmap="viridis", atom_size=120, vmax=None):
    """Filled contour plot of the surface over 2x2 cells with top-layer atoms, minima, saddles and path."""
    import matplotlib.pyplot as plt
    from ase.data.colors import jmol_colors
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    u = np.linspace(*window[0], n_fine)
    v = np.linspace(*window[1], n_fine)
    U, V = np.meshgrid(u, v, indexing="ij")
    XY = surf.origin + U[..., None] * surf.a1 + V[..., None] * surf.a2
    Z = surf(XY.reshape(-1, 2)).reshape(U.shape)
    cs = ax.contourf(XY[..., 0], XY[..., 1], Z, levels=levels, cmap=cmap, vmax=vmax)
    ax.contour(XY[..., 0], XY[..., 1], Z, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
    cb = plt.colorbar(cs, ax=ax)
    cb.set_label(r"$E - E_\mathrm{min}$ (eV)")
    # replicate points periodically over the window (fractional units)
    def images_in_window(xy):
        f0 = surf.Binv @ (np.asarray(xy)[:2] - surf.origin)
        out = []
        for n1 in range(int(np.floor(window[0][0] - f0[0])) - 1, int(np.ceil(window[0][1] - f0[0])) + 2):
            for n2 in range(int(np.floor(window[1][0] - f0[1])) - 1, int(np.ceil(window[1][1] - f0[1])) + 2):
                f = f0 + (n1, n2)
                if window[0][0] - 1e-6 <= f[0] <= window[0][1] + 1e-6 and \
                   window[1][0] - 1e-6 <= f[1] <= window[1][1] + 1e-6:
                    out.append(surf.origin + f[0] * surf.a1 + f[1] * surf.a2)
        return out
    for at in prim_top:
        for p in images_in_window(at.position):
            ax.scatter(*p, s=atom_size, color=jmol_colors[at.number], edgecolor="k", zorder=3)
    if minima is not None:
        for xy, _ in minima:
            for p in images_in_window(xy):
                ax.plot(*p, "o", mfc="w", mec="k", ms=7, zorder=4)
    if saddles is not None:
        for xy in saddles:
            ax.plot(*xy, "X", color="r", ms=10, mec="k", zorder=5)
    if path_xy is not None:
        ax.plot(path_xy[:, 0], path_xy[:, 1], "-", color="crimson", lw=2, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
    return ax
