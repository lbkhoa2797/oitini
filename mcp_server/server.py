"""MCP server exposing the materials-discovery tools.

Run with stdio transport:
    python -m mcp_server.server

Or expose over HTTP for remote clients:
    fastmcp run mcp_server/server.py --transport streamable-http
"""
import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from tools.literature import literature_search as _literature_search
from tools.materials_project import query_materials_project as _query_mp
from tools.dft import run_dft as _run_dft

mcp = FastMCP("materials-agent")


@mcp.tool()
def literature_search(query: str, top_k: int = 5,
                      section: Optional[str] = None) -> str:
    """Search a corpus of arXiv papers on 2D magnets and related materials.
    Returns chunks with chunk_id, arxiv_id, title, year, section, and text.
    Cite results by chunk_id when using them in answers.

    Args:
        query: Natural-language search query.
        top_k: How many chunks to return (1–10, default 5).
        section: Optional filter — one of 'abstract', 'introduction',
                 'methods', 'results', 'conclusion'.
    """
    hits = _literature_search(query=query, top_k=top_k, section=section)
    return json.dumps(hits, indent=2)


@mcp.tool()
def query_materials_project(formula: str) -> str:
    """Look up a material by chemical formula. Returns space group, band
    gap, magnetic state, and Curie temperature where known.

    Args:
        formula: e.g. 'Si', 'Fe', 'GaAs', 'NaCl'.
    """
    return json.dumps(_query_mp(formula), indent=2)


@mcp.tool()
def run_dft(formula: str, kgrid: list[int] | None = None,
            ecutwfc: float = 40.0, nspin: int = 1,
            magnetic_moments: dict[str, float] | list[float] | None = None,
            is_metal: bool = False, monolayer: bool = False,
            vacuum_A: float = 15.0) -> str:
    """Run a single-point DFT calculation via Quantum ESPRESSO. Returns
    total energy per atom (eV), SCF convergence, and magnetization.
    Cached by content hash; identical calls return instantly.

    Args:
        formula: Chemical formula; the bulk ground-state structure is fetched
            from Materials Project (e.g. Si, Fe, GaAs, NaCl).
        kgrid: Monkhorst-Pack k-point grid. Omit for an adaptive grid computed
            from the structure's reciprocal lattice at the default k-point
            spacing (DEFAULT_KPOINTS_DISTANCE in tools/grids.py); pass
            explicitly to override. Vacuum axes (e.g. the monolayer
            out-of-plane direction) are forced to 1 automatically.
        ecutwfc: Plane-wave cutoff in Ry (default 40, valid 30–120).
        nspin: 1 for non-magnetic, 2 for collinear spin-polarized.
        magnetic_moments: Initial local moments (muB) seeding the SCF. A per-element
            map for ferromagnets, e.g. {"Cr": 3.0}, or a per-atom list with opposite
            signs on the two sublattices for an antiferromagnet, e.g. [3.0, -3.0, ...].
            Setting this forces nspin=2.
        is_metal: True uses cold-smearing occupations (metals); False uses fixed
            occupations so QE reports the HOMO-LUMO gap (insulators/semiconductors).
        monolayer: True carves a single van der Waals layer out of the bulk
            structure and runs it as a 2D material with vacuum along z and
            QE's 2D Coulomb cutoff (assume_isolated='2D'). Fails for
            non-layered crystals.
        vacuum_A: Vacuum spacing (Angstrom) between periodic layer images when
            monolayer=True (default 15). May be raised automatically to satisfy
            the 2D Coulomb-cutoff cell constraints.
    """
    return json.dumps(_run_dft(formula, kgrid, ecutwfc, nspin, magnetic_moments,
                               is_metal=is_metal, monolayer=monolayer,
                               vacuum_A=vacuum_A),
                      indent=2, default=str)


@mcp.resource("materials://corpus/index")
def corpus_index() -> str:
    """Return a summary of the indexed literature corpus."""
    from pathlib import Path
    index = json.loads((Path("data") / "corpus_index.json").read_text())
    return json.dumps({
        "n_papers": len(index),
        "year_range": [min(r["year"] for r in index),
                       max(r["year"] for r in index)],
        "sample_titles": [r["title"] for r in index[:5]],
    }, indent=2)


@mcp.prompt()
def discover_2d_magnet(min_curie_K: int = 30,
                       min_bandgap_eV: float = 1.0) -> str:
    """A reusable prompt template for the 2D-magnet discovery task."""
    return (
        f"Find a 2D ferromagnetic semiconductor with Curie temperature "
        f"> {min_curie_K} K and direct bandgap > {min_bandgap_eV} eV. "
        f"Cite literature for any property values."
    )


if __name__ == "__main__":
    mcp.run()        # stdio transport by default
