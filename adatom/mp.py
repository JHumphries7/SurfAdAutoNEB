"""Materials Project access: structures and GGA (PBE) entries."""
import os
import sys

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

GGA = {"thermo_types": ["GGA_GGA+U"]}


def rester(api_key=None):
    from mp_api.client import MPRester
    key = api_key or os.environ.get("MP_API_KEY")
    if not key and os.path.exists("mp_api_key.txt"):
        key = open("mp_api_key.txt").read().strip()
    if not key:
        sys.exit("No Materials Project API key. Use --api-key, set MP_API_KEY, or put it in mp_api_key.txt")
    return MPRester(key)


def conventional(structure):
    return SpacegroupAnalyzer(structure, symprec=0.01).get_conventional_standard_structure()


def get_structure(mp_id, api_key=None):
    """PBE-relaxed structure of an MP entry, as the conventional cell."""
    with rester(api_key) as mpr:
        entries = mpr.get_entries(mp_id, additional_criteria=GGA)
    if not entries:
        sys.exit(f"No GGA (PBE) entry found for {mp_id}")
    return conventional(min(entries, key=lambda e: e.energy_per_atom).structure)


def get_chemsys_entries(elements, api_key=None):
    """All GGA entries (with structures) in the chemical system, e.g. ['Mo', 'C']."""
    with rester(api_key) as mpr:
        entries = mpr.get_entries_in_chemsys("-".join(elements), compatible_only=True,
                                             additional_criteria=GGA)
    for e in entries:                      # keep only the material id; the rest can't be saved to JSON
        e.data = {"material_id": str(e.data.get("material_id", ""))}
    return entries


def mp_id_of(entry):
    """Material id (e.g. mp-2305). The entry_id can be a calculation id instead."""
    mid = entry.data.get("material_id")
    return mid if mid else str(entry.entry_id).replace("-GGA+U", "").replace("-GGA", "")
