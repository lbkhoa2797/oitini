"""DFT via ASE + Quantum ESPRESSO. Single-point energy only, bulk or
carved monolayer (see monolayer=True); TODO = DFPT and relax structure
if needed."""
import hashlib
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
from langsmith import traceable
from pydantic import BaseModel, Field

from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.geometry import minkowski_reduce
from ase.geometry.dimensionality import isolate_components
from ase.units import Bohr

from config import DFT_RUNS_DIR, QE_BIN, MPI_NPROCS, QE_PSEUDOS_DIR
from tools.materials_project import fetch_summary_and_atoms
from tools.grids import DEFAULT_KPOINTS_DISTANCE, kgrid_from_spacing

# Cold-smearing width for is_metal runs (Ry): the Materials Cloud "medium"
# value, kept deliberately while DEFAULT_KPOINTS_DISTANCE sits at the coarse
# screening spacing.
METAL_DEGAUSS_RY = 0.02


# ---- input schema --------------------------------------------------------
class DFTInput(BaseModel):
    formula: str = Field(..., description="Bulk chemical formula, e.g. 'Si', 'Fe', 'GaAs', 'NaCl'.")
    kgrid: Optional[list[int]] = Field(
        default=None,
        description=(
            f"Monkhorst–Pack k-grid. Omit (default) for an adaptive grid computed "
            f"from the structure's reciprocal lattice at {DEFAULT_KPOINTS_DISTANCE} "
            "1/A k-point spacing (counts round up to even, 1 stays 1, so the "
            "derived phonon q-grid halves cleanly; vacuum axes are forced to 1). "
            "Pass an explicit grid only when the user or a convergence study "
            "calls for one — an explicit grid always wins."
        ),
    )
    monolayer: bool = Field(
        default=False,
        description=(
            "Set True for 2D materials: a single van der Waals layer is carved "
            "out of the bulk Materials Project structure, vacuum_A of vacuum is "
            "added along z, kgrid[2] is forced to 1, and QE's 2D Coulomb cutoff "
            "(assume_isolated='2D') is applied. Fails with an error for "
            "non-layered crystals. Note that Materials Project reference "
            "properties (band gap, magnetic ordering) describe the BULK crystal, "
            "not the monolayer."
        ),
    )
    vacuum_A: float = Field(
        15.0, ge=8.0, le=40.0,
        description="Vacuum spacing (Angstrom) between periodic images of the "
                    "layer when monolayer=True. Ignored for bulk runs. May be "
                    "raised automatically so the cell satisfies the 2D Coulomb "
                    "cutoff constraint c >= max(2 x thickness estimate, 20 bohr).",
    )
    ecutwfc: float = Field(40.0, ge=30.0, le=120.0,
                           description="Plane-wave cutoff (Ry).")
    nspin: int = Field(1, ge=1, le=2,
                       description="1=non-mag, 2=collinear-mag.")
    magnetic_moments: Optional[dict[str, float] | list[float]] = Field(
        default=None,
        description=(
            "Initial local magnetic moments (muB) used as the SCF starting guess. "
            "Either a per-element map for ferro/ferrimagnets, e.g. {'Cr': 3.0} "
            "(every Cr aligned up), or a per-atom list of length nat for "
            "antiferromagnets, e.g. [3.0, -3.0, 0, 0] to put two Cr on opposite "
            "sublattices. Distinct (element, moment) pairs are written as separate "
            "QE species, each with its own starting_magnetization(i) -- which is "
            "exactly what AFM/ferrimagnetic orders require. Setting this forces "
            "nspin=2."
        ),
    )
    is_metal: bool = Field(
        default=False,
        description=(
            "If True, use smearing occupations (Marzari-Vanderbilt cold smearing, "
            f"degauss={METAL_DEGAUSS_RY} Ry) "
            "suitable for metals. If False (default), use fixed occupations for "
            "insulators/semiconductors. The band gap is extracted from the run's "
            "XML (eigenvalues + occupations) in either mode."
        ),
    )


# ---- structure construction ---------------------------------------------
def _build_atoms(formula: str) -> Atoms:
    """Fetch the ground-state bulk structure for *formula* from Materials Project."""
    meta, atoms = fetch_summary_and_atoms(formula)
    if atoms is None:
        raise ValueError(meta.get("error", f"no Materials Project structure for '{formula}'"))
    return atoms


# ---- 2D / monolayer construction ----------------------------------------
def _carve_monolayer(atoms: Atoms, vacuum: float) -> "tuple[Atoms, dict]":
    """Extract a single van der Waals layer from a layered bulk crystal.

    ASE's dimensionality analysis (Larsen-style RDA) splits the bonded network
    into components; a vdW-layered crystal yields one 2D component per layer
    in the cell, already rotated so the layer lies in the xy-plane. The
    component's third cell vector is meaningless (zero or arbitrary), so it is
    rebuilt as [0, 0, thickness + vacuum] with the slab centred along z.

    The in-plane geometry is inherited from the bulk crystal unrelaxed -- the
    usual first approximation for vdW materials (interlayer coupling barely
    changes the intralayer bonds). Raises ValueError for non-layered inputs.
    """
    components = isolate_components(atoms)
    layers = components.get("2D", [])
    if not layers:
        found = {dim: len(parts) for dim, parts in components.items()}
        raise ValueError(
            f"{atoms.get_chemical_formula()} is not a van der Waals layered "
            f"crystal (bonded components by dimensionality: {found}); "
            "monolayer=True requires a layered parent structure."
        )
    layer = layers[0]
    cell = np.array(layer.cell)
    if np.abs(cell[:2, 2]).max() > 1e-3:
        raise ValueError("isolated 2D component is not aligned with the xy-plane")
    z = layer.positions[:, 2]
    thickness = float(z.max() - z.min())

    # QE's assume_isolated='2D' truncates the Coulomb kernel at lz = c/2
    # (PW/src/Coul_cut_2D.f90), so c must exceed twice the layer thickness
    # INCLUDING electron tails, and never drop below 20 bohr (pseudopotential
    # radial grids are cut at rcut = 10 bohr, Modules/read_pseudo.f90). QE does
    # not validate any of this itself. Thickness estimate: the larger of the
    # bulk interlayer repeat distance and (atomic thickness + 10 bohr).
    in_plane_area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
    d_inter = atoms.get_volume() / (in_plane_area * len(layers))
    t_est = max(d_inter, thickness + 10 * Bohr)
    c_min = max(2.0 * t_est, 20 * Bohr)
    vacuum_note = None
    if thickness + vacuum < c_min:
        raised = c_min - thickness
        vacuum_note = (
            f"vacuum raised from {vacuum:g} to {raised:.3f} A so that "
            f"c >= max(2 x thickness estimate, 20 bohr) = {c_min:.3f} A, as "
            "required by QE's assume_isolated='2D' Coulomb cutoff"
        )
        vacuum = raised
    cell[2] = [0.0, 0.0, thickness + vacuum]
    # In-plane vectors can come back skewed (e.g. a+b instead of b);
    # Minkowski reduction restores the compact cell. Positions are Cartesian,
    # so re-expressing the lattice does not move any atom.
    cell, _ = minkowski_reduce(cell, pbc=np.array([True, True, False]))
    layer.set_cell(cell)
    layer.center(axis=2)
    layer.wrap(pbc=[True, True, False])
    layer.pbc = True   # QE input is always 3D-periodic; the vacuum isolates z
    info = {
        "layers_in_bulk_cell": len(layers),
        "layer_formula": layer.get_chemical_formula(),
        "thickness_A": round(thickness, 3),
        "vacuum_A": round(vacuum, 3),
        "interlayer_distance_A": round(d_inter, 3),
    }
    if vacuum_note:
        info["vacuum_note"] = vacuum_note
    return layer, info


def _vacuum_axes(atoms: Atoms, min_vacuum: float = 7.0) -> list[int]:
    """Cell axes containing >= *min_vacuum* Angstrom of empty space.

    The gap is measured between atomic positions (nuclei) with periodic
    wrap-around. Bulk layered crystals have interlayer position gaps of
    ~3-4 A (graphite 3.35, CrI3 ~4), so 7 A only flags true vacuum regions:
    slabs, monolayers, molecules-in-boxes."""
    frac = atoms.get_scaled_positions(wrap=True)
    cell = np.array(atoms.get_cell())
    volume = abs(np.linalg.det(cell))
    axes = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        height = volume / np.linalg.norm(np.cross(cell[j], cell[k]))
        f = np.sort(frac[:, i])
        largest_gap = np.diff(f, append=f[0] + 1.0).max()
        if largest_gap * height >= min_vacuum:
            axes.append(i)
    return axes


# ---- pseudopotentials ---------------------------------------------------
# config.py holds these as plain strings so users can paste a path in without
# worrying about types; every tool module wraps them itself (see
# `Path(QE_BIN)` in phonon.py, `Path(W90_DIR)` in wannier90.py). This module
# used to glob QE_PSEUDOS_DIR directly, which raised
# "AttributeError: 'str' object has no attribute 'glob'" on every candidate.
PSEUDO_DIR = Path(QE_PSEUDOS_DIR)


def preflight_dft() -> None:
    """Validate the DFT toolchain once, before the discovery loop starts.

    A missing pw.x or an unreadable pseudo directory cannot be fixed by trying a
    different material, but simulator_node catches failures per candidate and
    keeps going. So one config mistake previously surfaced as 22 identical
    failures spread over 5 iterations -- after 22 Materials Project round-trips
    and a full report pass on an empty result. Fail once instead, naming the
    config key that needs fixing.
    """
    # Unset comes first and by name: these now live in .env, and Path("") is
    # Path("."), which would otherwise surface as a baffling "no *.upf in .".
    for key, value in (("QE_BIN", QE_BIN), ("QE_PSEUDOS_DIR", QE_PSEUDOS_DIR)):
        if not value:
            raise ValueError(
                f"{key} is not set. Add it to your .env — see env.sample for the "
                "expected keys."
            )

    pw = Path(QE_BIN) / "pw.x"
    if not pw.exists():
        raise FileNotFoundError(
            f"QE_BIN is {QE_BIN!r} but {pw} does not exist. Point QE_BIN in .env "
            "at the Quantum ESPRESSO bin/ directory."
        )
    if not os.access(pw, os.X_OK):
        raise PermissionError(f"{pw} is not executable (QE_BIN in .env).")
    if not PSEUDO_DIR.is_dir():
        raise NotADirectoryError(
            f"QE_PSEUDOS_DIR is {QE_PSEUDOS_DIR!r}, which is not a directory. "
            "Point it in .env at your UPF pseudopotential directory."
        )
    if not any(PSEUDO_DIR.glob("*.upf")):
        raise FileNotFoundError(
            f"no *.upf files in QE_PSEUDOS_DIR ({PSEUDO_DIR}). run_dft looks for "
            "'<Symbol>.upf', e.g. Zn.upf."
        )


def _pseudo_map(atoms: Atoms) -> dict:
    """Map element symbols to pseudopotential filenames in QE_PSEUDOS_DIR."""
    by_symbol = {}
    for symbol in set(atoms.get_chemical_symbols()):
        candidates = list(PSEUDO_DIR.glob(f"{symbol}.upf"))
        if not candidates:
            raise FileNotFoundError(
                f"no pseudopotential for '{symbol}' in {PSEUDO_DIR}"
            )
        by_symbol[symbol] = candidates[0].name
    return by_symbol


def _z_valence(symbol: str) -> float:
    """Read z_valence from the UPF pseudopotential file for *symbol*."""
    candidates = list(PSEUDO_DIR.glob(f"{symbol}.upf"))
    if not candidates:
        raise FileNotFoundError(
            f"no pseudopotential for '{symbol}' in {PSEUDO_DIR}"
        )
    text = candidates[0].read_text()
    m = re.search(r'z_valence\s*=\s*"?\s*([\d.]+)', text)
    if not m:
        raise ValueError(f"could not parse z_valence from {candidates[0]}")
    return float(m.group(1))


def _compute_nbnd(atoms: Atoms, nspin, extra: int = 8) -> int:
    """Compute nbnd = number of occupied bands + *extra*.

    Each band holds 2 electrons (nspin=1) or 1 electron per spin channel
    (nspin=2), but in both cases QE's occupied-band count equals
    ceil(nelec / 2), so the formula is the same.

    *extra* was 2 -- enough to locate the CBM but not to resolve the conduction
    band across the mesh, which _parse_qe_xml needs for cb_width_eV. Empty bands
    are cheap; 8 costs little and makes the conduction-band quantities meaningful.
    Note _cache_key does not include nbnd, so changing this does not invalidate
    already-cached runs -- clear data/dft_runs to recompute them.
    """
    symbols = atoms.get_chemical_symbols()
    nelec = sum(_z_valence(s) for s in symbols)
    if nspin != 2:
        n_occ = math.ceil(nelec / 2)
    else:
        n_occ = math.ceil(nelec)
    return n_occ + extra


# ---- magnetic starting guess --------------------------------------------
def _resolve_magmoms(atoms: Atoms, spec: "dict[str, float] | list[float]") -> list[float]:
    """Turn a magnetic spec into a per-atom list of initial moments (muB).

    QE's starting_magnetization is per-species, so antiferromagnetic order is
    expressed by giving atoms of the *same* element opposite moments; ASE then
    splits them into distinct species (Fe/Fe1) each with its own
    starting_magnetization(i). See ase.io.espresso.write_espresso_in.

    - dict {element: moment}: same moment on every atom of that element
      (ferromagnetic / ferrimagnetic seed).
    - list[moment]: one moment per atom, in structure order; use opposite signs
      on the two sublattices for an antiferromagnet, e.g. [+m, -m, ...].
    """
    symbols = atoms.get_chemical_symbols()
    if isinstance(spec, dict):
        return [float(spec.get(s, 0.0)) for s in symbols]
    if len(spec) != len(symbols):
        raise ValueError(
            f"magnetic_moments has {len(spec)} entries but the structure has "
            f"{len(symbols)} atoms; pass one moment per atom (order: "
            f"{symbols})."
        )
    return [float(x) for x in spec]


# ---- content-hash cache -------------------------------------------------
def _cache_key(formula: str, kgrid: list[int], ecutwfc: float, nspin: int,
               magnetic_moments=None, is_metal: bool = False,
               monolayer: bool = False, vacuum_A: Optional[float] = None) -> str:
    payload = [formula, kgrid, ecutwfc, nspin, magnetic_moments, is_metal,
               monolayer, vacuum_A if monolayer else None]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ---- output parser ------------------------------------------------------
HARTREE_EV = 27.211386245988   # CODATA; QE's XML stores energies in Hartree


def _find_qe_xml(workdir: Path) -> Optional[Path]:
    """Locate a completed run's data-file-schema XML: tmp/<prefix>.xml, with
    the copy inside tmp/<prefix>.save/ as fallback."""
    xmls = sorted((Path(workdir) / "tmp").glob("*.xml"))
    if xmls:
        return xmls[0]
    saves = sorted((Path(workdir) / "tmp").glob("*.save/data-file-schema.xml"))
    return saves[0] if saves else None


def _parse_qe_xml(xml_path: Path) -> dict:
    """Parse convergence, energy, magnetization, and band edges from pw.x's XML.

    The XML is the machine-readable record: unlike stdout it always carries
    every eigenvalue and occupation, so band edges are available for any
    occupation scheme -- fixed, smearing (where stdout prints only a Fermi
    energy), and spin-polarized (per-k arrays concatenate the up and down
    channels with occupations normalized to [0, 1], so one flat scan covers
    both channels).
    """
    out = ET.parse(xml_path).getroot().find("output")
    if out is None:
        return {}
    fields = {}

    conv = out.findtext("convergence_info/scf_conv/convergence_achieved")
    if conv is not None:
        fields["converged"] = conv.strip().lower() == "true"
    etot = out.findtext("total_energy/etot")
    if etot is not None:
        fields["total_energy_eV"] = float(etot) * HARTREE_EV

    mag = out.find("magnetization")
    if mag is not None:
        lsda = (mag.findtext("lsda") or "").strip().lower() == "true"
        total = mag.findtext("total")
        if total is not None:
            fields["total_magnetization"] = float(total)
        elif not lsda:
            fields["total_magnetization"] = 0.0   # nspin=1: zero by construction
        absolute = mag.findtext("absolute")
        if absolute is not None:
            fields["absolute_magnetization"] = float(absolute)

    bs = out.find("band_structure")
    if bs is None:
        return fields
    fermi = bs.findtext("fermi_energy")
    if fermi is not None:
        fields["fermi_energy_eV"] = float(fermi) * HARTREE_EV

    # Band edges by occupation threshold (occupied = occ > 0.5; smearing keeps
    # gapped states essentially at 0/1, and cold smearing's slight over/under-
    # shoots near the Fermi level never cross 0.5 the wrong way).
    vbm = cbm = direct = None
    homo_ks: list[float] = []   # per-k highest occupied -- valence dispersion
    lumo_ks: list[float] = []   # per-k lowest unoccupied -- conduction dispersion
    fractional = False
    for ks in bs.findall("ks_energies"):
        eig_el, occ_el = ks.find("eigenvalues"), ks.find("occupations")
        if eig_el is None or occ_el is None or not eig_el.text or not occ_el.text:
            continue
        eigs = [float(x) for x in eig_el.text.split()]
        occs = [float(x) for x in occ_el.text.split()]
        # Partially filled states = bands crossing the Fermi level. This is the
        # grid-robust metal test: on a coarse k-mesh a metal's eigenvalues near
        # E_F can be far apart (spurious "gap" between grid points), but their
        # occupations are fractional; a gapped system's stay at ~0/~1 whenever
        # gap >> degauss. Fixed-occupation runs have exact 0/1 (never trips).
        if not fractional and any(0.05 < o < 0.95 for o in occs):
            fractional = True
        k_occ = [e for e, o in zip(eigs, occs) if o > 0.5]
        k_uno = [e for e, o in zip(eigs, occs) if o <= 0.5]
        if k_occ:
            homo_ks.append(max(k_occ))
            vbm = max(k_occ) if vbm is None else max(vbm, max(k_occ))
        if k_uno:
            lumo_ks.append(min(k_uno))
            cbm = min(k_uno) if cbm is None else min(cbm, min(k_uno))
        if k_occ and k_uno:
            dk = min(k_uno) - max(k_occ)
            direct = dk if direct is None else min(direct, dk)

    if vbm is not None and cbm is not None:
        gap = (cbm - vbm) * HARTREE_EV
        # Metal = fractional occupations, or a negative/~zero gap (band overlap
        # in a wrongly-fixed run): report 0.0, never a negative or spurious gap.
        metallic = fractional or gap < 0.01
        fields.update({
            "homo_eV": vbm * HARTREE_EV,
            "lumo_eV": cbm * HARTREE_EV,
            "band_gap_eV": 0.0 if metallic else gap,
            "is_metallic": metallic,
            "band_gap_method": "xml_occupations",
        })
        if direct is not None:
            fields["band_gap_direct_eV"] = 0.0 if metallic else direct * HARTREE_EV
            # Whether the fundamental gap IS the optical one. Transparency is set
            # by the absorption onset and carrier generation by the fundamental
            # gap; in oxide semiconductors those routinely differ, so screening
            # both against one number produces false rejections.
            fields["gap_is_direct"] = bool(
                not metallic and (direct - (cbm - vbm)) * HARTREE_EV < 0.05)
        # Dispersion of the frontier bands across the sampled BZ. A conduction
        # band built from extended metal ns orbitals is wide (high mobility);
        # a flat one is not. Caveats: measured on the *coarse SCF k-grid*, so
        # these are proxies rather than converged bandwidths, and under nspin=2
        # the per-k arrays concatenate both spin channels, so a width spans both.
        if len(lumo_ks) > 1:
            fields["cb_width_eV"] = (max(lumo_ks) - min(lumo_ks)) * HARTREE_EV
        if len(homo_ks) > 1:
            fields["vb_width_eV"] = (max(homo_ks) - min(homo_ks)) * HARTREE_EV
    else:
        # No empty (or no filled) bands in the run -- fall back to the levels
        # QE itself reports for fixed-occupation calculations.
        homo = bs.findtext("highestOccupiedLevel")
        lumo = bs.findtext("lowestUnoccupiedLevel")
        if homo is not None and lumo is not None:
            gap = (float(lumo) - float(homo)) * HARTREE_EV
            metallic = gap < 0.01
            fields.update({
                "homo_eV": float(homo) * HARTREE_EV,
                "lumo_eV": float(lumo) * HARTREE_EV,
                "band_gap_eV": 0.0 if metallic else gap,
                "is_metallic": metallic,
                "band_gap_method": "xml_qe_levels",
            })
    return fields


def _parse_qe_output(workdir: Path) -> dict:
    """Pull a few key numbers out of pw.x output. ASE does most of this, but
    grabbing the convergence flag and any magnetic moment is easier by regex."""
    out = (workdir / "espresso.pwo").read_text()
    converged = "convergence has been achieved" in out
    energy = None
    m = re.search(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", out)
    if m:
        energy = float(m.group(1)) * 13.605698     # Ry → eV
    mag = None
    m = re.search(r"total magnetization\s+=\s+([-\d.]+)\s+Bohr", out)
    if m:
        mag = float(m.group(1))
    # Absolute magnetization is the meaningful number for AFM (total ~ 0).
    mabs = None
    m = re.search(r"absolute magnetization\s+=\s+([-\d.]+)\s+Bohr", out)
    if m:
        mabs = float(m.group(1))
    # Fixed-occupation runs print "highest occupied, lowest unoccupied level"
    # giving the HOMO-LUMO (KS) gap at the SCF k-mesh.
    homo = lumo = band_gap_eV = None
    m = re.search(
        r"highest occupied, lowest unoccupied level\s*\(ev\)\s*:\s*([-\d.]+)\s+([-\d.]+)",
        out, re.IGNORECASE,
    )
    if m:
        homo, lumo = float(m.group(1)), float(m.group(2))
        band_gap_eV = lumo - homo

    return {
        "converged": converged,
        "total_energy_eV": energy,
        "total_magnetization": mag,
        "absolute_magnetization": mabs,
        "homo_eV": homo,
        "lumo_eV": lumo,
        "band_gap_eV": band_gap_eV,
    }


# ---- tool spec ----------------------------------------------------------
TOOL_SPEC = {
    "name": "run_dft",
    "description": (
        "Run a single-point DFT calculation using Quantum ESPRESSO via ASE. "
        "Returns total energy (eV and eV/atom), convergence status, band gap "
        "(indirect and direct, with an is_metallic flag) and Fermi energy from "
        "the QE XML, and magnetic moment. Results are cached by input parameters. "
        "The k-grid defaults adaptively from the structure's reciprocal lattice "
        f"({DEFAULT_KPOINTS_DISTANCE} 1/A k-point spacing); pass kgrid only to "
        "override it. Use for accurate "
        "ground-state energies of materials; prefer this over empirical estimates. "
        "Supports 2D materials via monolayer=True: one van der Waals layer is "
        "carved from the bulk structure, vacuum is added, the k-grid becomes "
        "in-plane only ([n, n, 1]), and QE's 2D Coulomb cutoff "
        "(assume_isolated='2D') is applied. The returned cache_key is required "
        "by run_wannier90."
    ),
    "input_schema": DFTInput.model_json_schema(),
}


# ---- the tool function --------------------------------------------------
def _dft_inputs_for_trace(inputs: dict) -> dict:
    """Trace-only: log the ASE Atoms input as a short summary instead of an object dump."""
    atoms = inputs.get("atoms")
    if atoms is not None:
        inputs = {**inputs,
                  "atoms": f"{atoms.get_chemical_formula()} ({len(atoms)} atoms)"}
    return inputs


@traceable(run_type="tool", process_inputs=_dft_inputs_for_trace)
def run_dft(formula: str, kgrid: Optional[list[int]] = None,
            ecutwfc: float = 40.0, nspin: int = 1,
            magnetic_moments=None, atoms: Optional[Atoms] = None,
            is_metal: bool = False, monolayer: bool = False,
            vacuum_A: float = 15.0) -> dict:
    """Run a single-point QE calculation. Returns parsed properties.
    Cached by content hash of (formula, resolved kgrid, ecutwfc, nspin,
    magnetic_moments, is_metal, monolayer, vacuum_A-if-monolayer).

    kgrid=None (the default) resolves to an adaptive Monkhorst-Pack grid from
    the structure's reciprocal lattice at DEFAULT_KPOINTS_DISTANCE (see
    tools/grids.py), computed on the final cell (the carved layer for monolayer
    runs): counts round up to even (1 stays 1) so the default phonon q-grid
    halves cleanly, and vacuum axes are forced to 1. An explicit kgrid always
    wins (vacuum axes are still forced to 1). The grid actually used is echoed
    as kgrid/kgrid_source in the result.

    is_metal controls the occupation scheme:
      False (default) → occupations='fixed' (insulator/semiconductor).
      True → occupations='smearing' with Marzari-Vanderbilt cold smearing
      (width METAL_DEGAUSS_RY).
    Band edges (band_gap_eV, band_gap_direct_eV, homo/lumo, fermi_energy_eV,
    is_metallic) come from the QE XML's eigenvalues + occupations, so they are
    available under both schemes; stdout parsing is only a fallback.

    magnetic_moments seeds the SCF spin density (see DFTInput): a per-element
    dict for ferro/ferrimagnets, or a per-atom list with opposite signs on the
    two sublattices for an antiferromagnet. NOTE: with monolayer=True a
    per-atom list refers to the carved layer, not the bulk cell.

    monolayer=True carves one vdW layer out of the (bulk) structure via
    _carve_monolayer, adds vacuum_A of vacuum along z, forces kgrid[2]=1 and
    applies QE's 2D Coulomb cutoff (assume_isolated='2D'). The cutoff requires
    c >= max(2 x layer thickness estimate, 20 bohr); vacuum_A is raised
    automatically when the requested value is too small (reported in
    layer_info.vacuum_note). Monolayer runs live under
    DFT_RUNS_DIR/{formula}_2d_{key} (bulk: {formula}_{key}). The QE prefix is
    set to the same label, so outputs are tmp/{label}.xml and tmp/{label}.save.

    atoms, if given, is the pre-fetched structure for *formula* (must correspond
    to it -- the cache is still keyed on formula); when omitted it is fetched
    from Materials Project. Lets callers avoid a duplicate MP lookup."""
    # The structure comes first: the adaptive default needs the (carved) cell,
    # and the cache key must hash the grid actually run.
    kgrid_source = ("user" if kgrid is not None
                    else f"adaptive ({DEFAULT_KPOINTS_DISTANCE} 1/A)")
    atoms = atoms if atoms is not None else _build_atoms(formula)
    layer_info = None
    if monolayer:
        try:
            atoms, layer_info = _carve_monolayer(atoms, vacuum_A)
        except ValueError as e:
            return {"dft_status": "failed", "error": str(e), "formula": formula}
    kgrid = list(kgrid) if kgrid is not None else kgrid_from_spacing(atoms)
    # Guard: never sample k along a vacuum direction (monolayers after carving,
    # slabs handed in via *atoms*, oddball MP entries).
    kgrid_note = None
    vacuum_dirs = [i for i in _vacuum_axes(atoms) if kgrid[i] > 1]
    if vacuum_dirs:
        for i in vacuum_dirs:
            kgrid[i] = 1
        if kgrid_source == "user":
            kgrid_note = (f"kgrid forced to 1 along vacuum axis(es) {vacuum_dirs}: "
                          "the structure has no periodic images to disperse across there")

    key = _cache_key(formula, kgrid, ecutwfc, nspin, magnetic_moments, is_metal,
                     monolayer, vacuum_A)
    label = f"{formula}_2d" if monolayer else formula
    workdir = DFT_RUNS_DIR / f"{label}_{key}"
    cache_file = workdir / "result.json"
    if cache_file.exists():
        return {**json.loads(cache_file.read_text()), "dft_status": "cached"}

    workdir.mkdir(parents=True, exist_ok=True)
    pseudos = _pseudo_map(atoms)

    # Magnetic starting guess. starting_magnetization is per-species in QE, so
    # we set per-atom moments and let ASE generate the species split + the
    # indexed starting_magnetization(i) (handles FM, AFM, ferrimagnetic alike).
    if magnetic_moments is None and nspin == 2:
        # Back-compat default: uniform ferromagnetic seed on every species.
        magnetic_moments = {s: 1.0 for s in set(atoms.get_chemical_symbols())}
    if magnetic_moments is not None:
        atoms.set_initial_magnetic_moments(_resolve_magmoms(atoms, magnetic_moments))
        nspin = 2  # any nonzero moment requires collinear spin-polarised SCF

    profile = EspressoProfile(
        command=f"mpirun -n {MPI_NPROCS} {QE_BIN}/pw.x",
        pseudo_dir=str(QE_PSEUDOS_DIR),
    )

    nbnd = _compute_nbnd(atoms, nspin)
    system_block = {"ecutwfc": ecutwfc, "ecutrho": ecutwfc * 4.0, "nspin": nspin,
                    "nbnd": nbnd}
    if monolayer:
        # Sohier's 2D Coulomb cutoff -- kills spurious electrostatic coupling
        # between periodic images. Geometry constraints (c >= max(2 x thickness
        # estimate, 20 bohr)) are enforced in _carve_monolayer.
        system_block["assume_isolated"] = "2D"
    if is_metal:
        system_block.update({"occupations": "smearing", "smearing": "mv",
                             "degauss": METAL_DEGAUSS_RY})
    else:
        system_block["occupations"] = "fixed"
        if nspin == 2:
            # Fixed occupations + nspin=2 requires tot_magnetization so QE
            # knows how to split electrons between spin channels.  Compute it
            # from the initial magnetic moments already set on the atoms.
            tot_mag = sum(atoms.get_initial_magnetic_moments())
            system_block["tot_magnetization"] = round(tot_mag)

    input_data = {
        # prefix = the dir label ({formula} or {formula}_2d) so QE outputs are
        # tmp/{label}.xml / tmp/{label}.save, matching the Wannier90 seed.
        "control": {"calculation": "scf", "verbosity": "high",
                    "prefix": label,
                    "outdir": str(workdir / "tmp")},
        # starting_magnetization is injected per species by ASE from the atoms'
        # initial_magnetic_moments set above -- do not hardcode it here.
        "system": system_block,
        "electrons": {"conv_thr": 1e-10, "mixing_beta": 0.4, "electron_maxstep": 200},
    }
    calc = Espresso(profile=profile, pseudopotentials=pseudos,
                    kpts=kgrid, input_data=input_data,
                    directory=str(workdir))

    atoms.calc = calc
    try:
        atoms.get_potential_energy()         # blocks until pw.x exits
    except Exception as e:
        return {"dft_status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "workdir": str(workdir)}

    parsed = _parse_qe_output(workdir)
    if parsed.get("band_gap_eV") is not None:
        parsed["band_gap_method"] = "stdout"
    # XML values win when available; the stdout regexes above are only the
    # fallback for a missing/corrupt XML.
    try:
        xml_path = _find_qe_xml(workdir)
        if xml_path is not None:
            parsed.update(
                {k: v for k, v in _parse_qe_xml(xml_path).items() if v is not None})
    except Exception:
        pass
    n_atoms = len(atoms)
    parsed["total_energy_eV_per_atom"] = (
        parsed["total_energy_eV"] / n_atoms if parsed["total_energy_eV"] else None
    )
    parsed["formula"] = formula
    parsed["kgrid"] = kgrid
    parsed["kgrid_source"] = kgrid_source
    parsed["ecutwfc"] = ecutwfc
    parsed["nspin"] = nspin
    parsed["magnetic_moments"] = magnetic_moments
    parsed["is_metal"] = is_metal
    if is_metal:
        parsed["degauss"] = METAL_DEGAUSS_RY
    parsed["monolayer"] = monolayer
    if monolayer:
        parsed["assume_isolated"] = "2D"
    if layer_info is not None:
        parsed["layer_info"] = layer_info
    if kgrid_note is not None:
        parsed["kgrid_note"] = kgrid_note
    parsed["workdir"] = str(workdir)
    parsed["cache_key"] = key
    parsed["dft_status"] = "done" if parsed["converged"] else "failed"

    # Include band info so the agent can reason about nbnd for NSCF
    from tools.wannier90 import parse_scf_band_info
    try:
        parsed["band_info"] = parse_scf_band_info(workdir)
    except Exception:
        pass

    cache_file.write_text(json.dumps(parsed, indent=2, default=str))
    return parsed
