"""The machine-learned potential. Add your own backend here if needed."""


def get_calc(backend):
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
