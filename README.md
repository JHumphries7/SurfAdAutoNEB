# SurfAdAutoNEB

Step-by-step scripts to estimate how easily surface atoms move on a crystalline compound:

- **\E_f**: energy to create a surface adatom from the material itself
- **E_m**: barrier for that adatom to migrate along a chosen direction
- **E_A = E_f + E_m**: the activation energy to compare between materials

A machine-learned potential (MLIP) finds the adatom sites and the migration path.
Quantum ESPRESSO (QE) then gives the energies of the minima and the saddle point.
Each script either generates input files for you to run, or is itself the job to run.

## How it works

| Step | Script | What it does | What you run afterwards |
|---|---|---|---|
| 1 | `01_setup.py` | Gets the conventional cell, writes `config.yaml` | – |
| 2 | `02_bulk_phases.py` | Picks the bulk phases needed for the chemical potentials, writes their SCF inputs | `pw.x` on `bulk/*_scf.pwi` |
| 3 | `03_chempots.py` | Reads the bulk energies, writes `chempots.json` | – |
| 4 | `04_choose_cases.py` | Lists facets, terminations, adatom species and directions; you choose; builds slabs and clean-slab inputs | `pw.x` on `dft/<slab>/*_clean_relax.pwi` |
| 5 | `05_pes.py` | Energy map of the adatom over the surface (MLIP) | this script, on a GPU node |
| 6 | `06_path.py` | Finds and plots the migration path on the map | – (look at the plot) |
| 7 | `07_neb.py` | Climbing-image NEB along the path (MLIP) | this script, on a GPU node |
| 8 | `08_dft_inputs.py` | Writes QE inputs for the minima and saddle points | `pw.x` on `dft/<case>/*.pwi` |
| 9 | `09_results.py` | Collects E_f, E_m, E_A into a table and `results.csv` | – |
| 10 | `10_nebx_input.py` | Optional: `neb.x` input for a full DFT NEB check of one hop | `neb.x` |

Steps 2–3 (chemical potentials) and steps 4–8 (slabs and paths) are independent, so you can
run them in parallel. Step 9 needs both.

## Install

You need Python 3.10–3.12, Quantum ESPRESSO (`pw.x`; `neb.x` only for step 10) and a GPU for the MLIP steps.

```bash
git clone <your repository url> adatom-barriers
conda create -n adatom python=3.11
conda activate adatom
pip install fairchem-core==1.10.0          # the default MLIP (eSEN-30M-OAM); pulls torch 2.4 and numpy < 2
pip install -r adatom-barriers/requirements.txt
```

The default MLIP is **eSEN-30M-OAM** (`backend: esen_oam`). Its weights are gated: request access to
`facebook/OMAT24` on Hugging Face, then run `huggingface-cli login` once.

Other MLIPs: set `backend` in `config.yaml` (or `--backend` in step 1) and install the matching package.
Only steps 5 and 7 load the MLIP, so a second environment is only needed for those two scripts.

| `backend` | Install | Notes |
|---|---|---|
| `esen_oam` (default) | `pip install fairchem-core==1.10.0` (Python <= 3.12, `numpy<2`, `torch~=2.4.0`) | eSEN-30M-OAM. Best overall in a 2026 surface benchmark of universal MLIPs. |
| `uma` | `pip install fairchem-core` (2.x) in a **separate** environment | Meta's UMA model; a good cross-check. Request access to `facebook/UMA` on Hugging Face. |
| `mace` | `pip install mace-torch` | MACE-OMAT. |
| `emt` | nothing | ASE toy potential for Cu, Al, Ni, Pd, Ag, Pt, Au. Use it for the self-test below. |
| `test` | nothing | Lennard-Jones. Only checks that the scripts run; the numbers mean nothing. |

To use another model, add a few lines to `adatom/mlip.py`.

You also need:

- **A Materials Project API key** (free): log in at materialsproject.org and copy it from your dashboard's API page.
- **Pseudopotentials** named `<Element>.upf` (for example `Mo.upf`, `C.upf`) in one folder. Use the same set for every calculation.

## Self-test (no GPU, no QE, 5 minutes)

This runs steps 1 and 4–8 on copper with the toy `emt` potential:

```bash
mkdir cu_test && cd cu_test
python -c "from ase.build import bulk; bulk('Cu', 'fcc', a=3.61, cubic=True).write('Cu.cif')"
python ../adatom-barriers/01_setup.py --cif Cu.cif --backend emt
python ../adatom-barriers/04_choose_cases.py --facet 1 0 0 --terminations all --species all --directions a1
python ../adatom-barriers/05_pes.py --all
python ../adatom-barriers/06_path.py --all
python ../adatom-barriers/07_neb.py --all
python ../adatom-barriers/08_dft_inputs.py --all
```

Step 6 should report a barrier of about 0.42 eV, and `pes/*_route.png` should show the map and the path.

## Walkthrough

Make one folder per material and run every script from inside it. Below, `$AB` is the path to this repository.

### 1. Set up

```bash
mkdir MoC && cd MoC
python $AB/01_setup.py --mp-id mp-2305 --api-key YOUR_KEY --pseudo-dir /path/to/pseudos
```

This downloads the PBE-relaxed structure as the conventional cell, saves it to `structure/`, and writes `config.yaml`.
The API key is saved to `mp_api_key.txt` so later steps find it (you can instead `export MP_API_KEY=...`).

To use your own structure file: `--cif my.cif --mp-id mp-2305`. It is converted to the conventional cell
unless you add `--keep-cell`. The `--mp-id` is still needed for step 2.

**Open `config.yaml` and check it.** Every setting used later lives there:

| Section | Setting | Meaning |
|---|---|---|
| `qe` | `pseudo_dir`, `ecutwfc`, `ecutrho`, `smearing`, `degauss`, `nspin` | Written into every QE input |
| `qe` | `kspacing` | k-point spacing in 1/Å; sets the k-mesh of every bulk and slab |
| `qe` | `forc_conv_thr` | Force threshold for relaxations (Ry/Bohr; 3.9e-4 = 0.01 eV/Å) |
| `slab` | `min_thickness` or `n_planes` | Slab thickness in Å, or an exact number of atomic planes |
| `slab` | `vacuum` | Total vacuum, Å |
| `slab` | `min_width` or `repeat` | In-plane supercell: at least this wide in Å, or e.g. `[4, 4]` |
| `mlip` | `backend` | Which MLIP to use |
| `mlip` | `relax_substrate` | Relax the upper half of the slab in the MLIP steps (keep `true` for real runs) |
| `pes` | `grid_spacing` | Spacing of the energy-map grid, Å |
| `neb` | `n_images`, `fmax` | NEB images (including end points) and force threshold |

Anything not listed (mixing, convergence thresholds, …) you can edit directly in the generated QE inputs.

### 2. Bulk phases for the chemical potentials

```bash
python $AB/02_bulk_phases.py
```

This builds the Materials Project PBE convex hull around your material and picks the phases to compute:

- `host`: your material
- `vertex`: phases that coexist with it at the corners of its stability region
- `elem_ref` / `elem_alt`: the lowest-energy form of each element, plus close competitors
- `near_hull`: nearby compounds that could change the hull with your own energies

Gas elements (H, N, O, F, Cl) are computed as an isolated molecule in a box.
The script writes one structure and one SCF input per phase in `bulk/`. Run each one:

```bash
cd bulk
pw.x -in MoC_host_scf.pwi > MoC_host_scf.pwo
```

**Outputs must have the same name as the input, ending in `.pwo`.** This holds for every QE run in this workflow.

### 3. Chemical potentials

```bash
python $AB/03_chempots.py
```

This rebuilds the hull from *your* energies and prints each vertex with its chemical potentials.
If your material is slightly above the hull, it is shifted onto it and the script says so.
The reference vertex is the metal-rich one (lowest chemical potential of the most electronegative element).
Use `--vertex N` to pick another. The result is saved in `chempots.json`.

Check the printed pressures: a few tens of kbar is fine for a structure that was not re-relaxed with your settings.

### 4. Choose what to compute

```bash
python $AB/04_choose_cases.py --list     # show every option
python $AB/04_choose_cases.py            # choose interactively
```

The script walks through four numbered lists and asks you to pick from each (`1,3`, `2-4` or `all`):

1. **Facets**: symmetry-distinct Miller indices of the conventional cell (`--max-index 2` for more).
2. **Terminations** of each facet. Each line shows the top atomic plane, the plane below it, and how many
   bonds the surface atoms keep. The ones with the fewest broken bonds are marked; those are usually the stable cuts.
3. **Adatom species**.
4. **Migration directions**: the two in-plane lattice vectors of the slab, with their crystal directions.
   Pick the one along your transport direction.

Or give everything on the command line:

```bash
python $AB/04_choose_cases.py --facet 1 0 0 --terminations 2,3 --species Mo,C --directions a2
```

Each combination becomes a **case** with an id like `100_t2_Mo__C__a2`
(facet 100, termination 2 with Mo on top, C adatom, moving along a2). Later scripts take case ids,
case numbers, or `--all`. You can run this step again to add more cases.

The script writes the slabs to `slabs/` and a clean-slab relaxation to `dft/<slab>/`. Run those with `pw.x`.
The bottom half of each slab is fixed and a dipole correction is included.

Useful options: `--planes 12` (exact number of atomic planes), `--thickness 15`, `--repeat 4 4`.

### 5. Energy map (GPU)

```bash
python $AB/05_pes.py --all
```

For each slab and adatom species, the adatom is placed on a grid over the surface cell. At each point its
x and y are held fixed while its height and the slab relax. Symmetry is used to skip equivalent points.
Submit this as a job on a GPU node. Finished maps are skipped on a rerun (`--redo` forces them).

### 6. Migration path

```bash
python $AB/06_path.py --all
```

This finds the lowest-barrier route from the best adatom site to the same site one lattice vector along the
chosen direction. It writes `pes/<case>_route.png`. **Look at it** before going on:

- top left: the energy map with surface atoms, minima (white dots), saddle points (red crosses) and the route
- top right: energy along the route
- bottom left: adatom height
- bottom right: problem points (should be empty)

If the route passes through intermediate minima, it is split into several hops.

### 7. NEB (GPU)

```bash
python $AB/07_neb.py --all
```

This relaxes the minima, runs a climbing-image NEB for each hop, and saves the start, saddle and end
structures in `neb/`. It prints the MLIP barrier.

### 8. DFT inputs for minima and saddles

```bash
python $AB/08_dft_inputs.py --all
```

For each hop this writes, in `dft/<case>/`:

- `*_min<h>_relax.pwi`: relaxation of the minimum
- `*_ts<h>_relax.pwi`: the saddle point, relaxed with only the adatom's x and y fixed

Run them all with `pw.x`. With `--single-point` it also writes `*_ts<h>_scf.pwi`, a cheaper single point at the
MLIP saddle geometry (less accurate; use `09_results.py --single-point` to read those instead).

### 9. Results

```bash
python $AB/09_results.py
```

For each case:

- E_f = E(slab + adatom, lowest minimum) − E(clean slab) − μ(adatom species)
- E_m = E(highest saddle on the route) − E(lowest minimum)
- E_A = E_f + E_m

The table also shows the MLIP barrier and which runs are still missing. Results go to `results.csv` (eV).
The lowest E_A over all cases is the material's value.

### 10. Optional: full DFT NEB check

```bash
python $AB/10_nebx_input.py 100_t2_Mo__C__a2 --hop 0
neb.x -inp dft/100_t2_Mo__C__a2/100_t2_Mo__C__a2_hop0_neb.in
```

Run this once on the hop that sets the lowest E_A to check that the cheaper route agrees.

## Files in a project folder

```
config.yaml        settings                      cases.json     chosen slabs and cases
structure/         conventional cell             chempots.json  chemical potentials
bulk/              bulk phases, inputs, outputs  results.csv    final table
slabs/             slab structures               pes/           energy maps, routes, plots
dft/<slab>/        clean-slab relaxation         neb/           MLIP NEB structures
dft/<case>/        minima and saddle inputs
```

## Things to know

- **Untested parts.** The scripts were tested end to end with toy potentials and synthetic QE outputs.
  The Materials Project download, the `esen_oam` / `uma` / `mace` backends, parsing of real QE output, and the
  `neb.x` input have not been run for real yet. Check the first real run of each carefully.
- **Slab size drives the cost.** Check the atom count printed in step 4 before submitting DFT jobs.
- **The map misses exchange mechanisms**, where the adatom swaps places with a surface atom. If the plot in
  step 6 shows flagged points or the adatom height drops sharply, look at that case by hand.
- **Spin-orbit coupling is not included.** Inputs are collinear spin-polarized (`nspin: 2`) or non-magnetic (`nspin: 1`).
- **Facets with very closely spaced planes** may need a different `plane_tol` in `config.yaml`.
- **Do not mix settings.** Every energy in E_f comes from a different run, so cutoff, pseudopotentials,
  smearing and spin must be the same everywhere. Change them in `config.yaml` before step 2, not afterwards.
