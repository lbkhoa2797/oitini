from typing import Optional, cast

from langsmith import traceable
from pydantic import BaseModel, Field
from mp_api.client import MPRester
from ase import Atoms
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor

from config import MP_API_KEY


class MPQueryInput(BaseModel):
    formula: str = Field(..., description="The chemical formula of the material, e.g. 'CrSBr'.")


# Summary fields pulled for every lookup; "structure" is added on demand.
_SUMMARY_FIELDS = [
    "material_id", "formula_pretty", "energy_above_hull",
    "formation_energy_per_atom", "band_gap", "is_magnetic", "ordering",
    "total_magnetization", "symmetry", "nsites", "density",
]


def _formula_error(formula: str) -> Optional[str]:
    """Error message if *formula* is not a plain stoichiometric formula, else None."""
    try:
        Composition(formula)
        return None
    except Exception:
        return (
            f"'{formula}' is not a valid stoichiometric chemical formula. "
            "Use a plain formula like 'ZnO' or 'In2O3' — no dopant notation, "
            "percentages, or prose."
        )


def _best_doc(formula: str, fields: list[str]):
    """Most stable Materials Project summary doc for *formula*, or None."""
    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.materials.summary.search(formula=formula, fields=fields)
    return min(docs, key=lambda d: d.energy_above_hull) if docs else None


def _summary_dict(doc) -> dict:
    """Map an MP summary doc to our JSON-safe property dict."""
    return {
        "material_id": doc.material_id,
        "formula": doc.formula_pretty,
        "space_group": doc.symmetry.symbol if doc.symmetry else None,
        "band_gap_eV": doc.band_gap,
        "is_magnetic": doc.is_magnetic,
        "magnetic_ordering": str(doc.ordering) if doc.ordering else None,
        "total_magnetization_uB": doc.total_magnetization,
        "energy_above_hull_eV_per_atom": doc.energy_above_hull,
        "formation_energy_eV_per_atom": doc.formation_energy_per_atom,
        "nsites": doc.nsites,
        "density_g_cm3": doc.density,
    }


def query_materials_project(formula: str) -> dict:
    """Look up a material by formula via the real Materials Project API.
    Returns ground-state properties for the most stable entry."""
    if not MP_API_KEY:
        return {"error": "MP_API_KEY is not set. Add it to your .env file."}
    if err := _formula_error(formula):
        return {"error": err}
    try:
        doc = _best_doc(formula, _SUMMARY_FIELDS)
    except Exception as e:
        return {"error": f"Materials Project query failed: {type(e).__name__}: {e}"}
    if doc is None:
        return {"error": f"No Materials Project entries found for '{formula}'."}
    return _summary_dict(doc)


def _mp_output_for_trace(out: tuple) -> dict:
    """Trace-only: log the ASE Atoms as a short summary instead of an object dump."""
    summary, atoms = out
    return {
        "summary": summary,
        "atoms": None if atoms is None
        else f"{atoms.get_chemical_formula()} ({len(atoms)} atoms)",
    }


@traceable(run_type="tool", process_outputs=_mp_output_for_trace)
def fetch_summary_and_atoms(formula: str) -> tuple[dict, Optional[Atoms]]:
    """One MP round-trip returning the same summary dict as
    query_materials_project plus the ground-state structure as ASE Atoms.
    On any failure the dict carries an "error" key and Atoms is None."""
    if not MP_API_KEY:
        return {"error": "MP_API_KEY is not set. Add it to your .env file."}, None
    if err := _formula_error(formula):
        return {"error": err}, None
    try:
        doc = _best_doc(formula, _SUMMARY_FIELDS + ["structure"])
    except Exception as e:
        return {"error": f"Materials Project query failed: {type(e).__name__}: {e}"}, None
    if doc is None:
        return {"error": f"No Materials Project entries found for '{formula}'."}, None
    return _summary_dict(doc), cast(Atoms, AseAtomsAdaptor.get_atoms(doc.structure))


TOOL_SPEC = {
    "name": "query_materials_project",
    "description": (
        "Look up a material in the Materials Project database by chemical formula. "
        "Returns space group, band gap (eV), magnetic ordering, total magnetization, "
        "formation energy, and energy above hull for the most stable entry. "
        "Use this before running DFT to get a quick structural and electronic overview."
    ),
    "input_schema": MPQueryInput.model_json_schema(),
}
