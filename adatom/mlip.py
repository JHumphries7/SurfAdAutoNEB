"""The machine-learned potential. Add your own backend here if needed."""


def get_calc(backend, d3=None):
    """MLIP calculator; d3 = 'bj' or 'zero' adds the DFT-D3 correction (same as QE's vdw_corr)."""
    calc = _base_calc(backend)
    if not d3:
        return calc
    import torch
    from ase.units import Bohr
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    # 40 Bohr (21 A) cutoff: forces match the full 95 Bohr to ~1e-3 eV/A at a fraction of the memory
    return TorchDFTD3Calculator(dft=calc, damping=d3, xc="pbe", abc=False, dtype=torch.float64,
                                cutoff=40.0 * Bohr, cutoff_smoothing="poly",
                                device="cuda" if torch.cuda.is_available() else "cpu")


def d3_from_config(cfg):
    """'bj' / 'zero' if the QE inputs use D3, else None, so the MLIP matches the DFT."""
    vdw = (cfg["qe"].get("vdw_corr") or "").lower()
    if "d3" not in vdw:
        return None
    return "bj" if cfg["qe"].get("dftd3_version", 3) == 4 else "zero"


def _base_calc(backend):
    if backend == "uma":                   # fairchem-core 2.x
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        unit = pretrained_mlip.get_predict_unit("uma-s-1p2p1", device="cuda", inference_settings="high-precision")
        return FAIRChemCalculator(unit, task_name="omat")
    if backend == "esen_oam":              # fairchem-core 1.10 (separate environment)
        from fairchem.core import OCPCalculator
        return OCPCalculator(model_name="eSEN-30M-OAM", local_cache="./fairchem_cache", cpu=False, seed=0)
    if backend == "mace":                  # mace-torch
        from mace.calculators import mace_mp
        return mace_mp(model="medium-omat-0", default_dtype="float64", device="cuda")
    if backend == "emt":                   # ASE toy potential (Cu, Al, Ni, Pd, Ag, Pt, Au): quick self-test
        from ase.calculators.emt import EMT
        return EMT()
    if backend == "test":                  # Lennard-Jones: only to check that the scripts run
        from ase.calculators.lj import LennardJones
        return LennardJones(sigma=1.95, epsilon=0.3, rc=5.0, smooth=True)
    raise ValueError(f"unknown MLIP backend '{backend}' (uma, esen_oam, mace, emt, test)")
