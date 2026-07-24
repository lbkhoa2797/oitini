"""Wannier90 step (NSCF -> pw2wannier90 -> wannier90.x) on top of a completed
run_dft SCF calculation -- bulk or carved monolayer -- located by its cache_key."""
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal, Optional

from ase.io import read as ase_read
from pydantic import BaseModel, Field

from config import DFT_RUNS_DIR, MPI_NPROCS, QE_BIN, W90_DIR, W90_RUNS_DIR
from tools.dft import _vacuum_axes

KMESH_PL = Path(W90_DIR) / "utility" / "kmesh.pl"
W90_X = str(Path(W90_DIR) / "wannier90.x")
PW2WAN_X = str(Path(QE_BIN) / "pw2wannier90.x")

HARTREE_EV = 27.211386245988


def parse_scf_band_info(scf_workdir: Path) -> dict:
    """Parse nelec, nbnd, and element list from a completed SCF run's
    tmp/<prefix>.xml (run_dft sets prefix to the run's label)."""
    xmls = sorted((Path(scf_workdir) / "tmp").glob("*.xml"))
    xml_path = xmls[0] if xmls else Path(scf_workdir) / "tmp" / "missing.xml"
    if not xml_path.exists():
        # Fall back to the .save directory
        save_dirs = list((Path(scf_workdir) / "tmp").glob("*.save"))
        if save_dirs:
            xml_path = save_dirs[0] / "data-file-schema.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"No QE XML output found in {scf_workdir}/tmp/")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    bs = root.find(".//band_structure")
    if bs is None:
        raise ValueError(f"No <band_structure> section in {xml_path}")

    nelec = float(bs.findtext("nelec", "0"))
    lsda = bs.findtext("lsda", "false").strip().lower() == "true"

    if lsda:
        nbnd_up = int(bs.findtext("nbnd_up", "0"))
        nbnd_dw = int(bs.findtext("nbnd_dw", "0"))
        nbnd_scf = max(nbnd_up, nbnd_dw)
    else:
        nbnd_scf = int(bs.findtext("nbnd", "0"))

    num_atomic_wfc = int(bs.findtext("num_of_atomic_wfc", "0"))

    # Energy reference for the disentanglement windows. QE stores these in
    # Hartree; convert to eV (the same absolute reference as the .eig file the
    # Wannier90 step reads). For metals/smeared runs <fermi_energy> is present;
    # for fixed-occupation insulators QE writes the HOMO/LUMO levels instead.
    def _ev(tag: str) -> Optional[float]:
        txt = bs.findtext(tag)
        return float(txt) * HARTREE_EV if txt not in (None, "") else None

    fermi_energy_eV = _ev("fermi_energy")
    homo_eV = _ev("highestOccupiedLevel")
    lumo_eV = _ev("lowestUnoccupiedLevel")
    if fermi_energy_eV is None:
        fermi_energy_eV = homo_eV

    # Extract element symbols and atom count from the final structure. QE
    # writes <atomic_positions> under both <input> and <output>, so restrict
    # to <output> or every atom is counted twice.
    out_atoms = root.findall("./output//atomic_positions/atom")
    elements = sorted({
        atom.get("name", "").rstrip("0123456789") for atom in out_atoms
    })

    n_atoms = len(out_atoms)

    return {
        "nelec": nelec,
        "nbnd_scf": nbnd_scf,
        "num_atomic_wfc": num_atomic_wfc,
        "lsda": lsda,
        "elements": elements,
        "n_atoms": n_atoms,
        "fermi_energy_eV": fermi_energy_eV,
        "highest_occupied_eV": homo_eV,
        "lowest_unoccupied_eV": lumo_eV,
    }


def _kmesh_block(nk1: int, nk2: int, nk3: int) -> str:
    """Run wannier90's kmesh.pl to get an explicit K_POINTS crystal block."""
    result = subprocess.run(
        ["perl", str(KMESH_PL), str(nk1), str(nk2), str(nk3)],
        check=True, capture_output=True, text=True,
    )
    return result.stdout


def make_nscf_input(scf_workdir: Path, nscf_workdir: Path,
                    nscf_kgrid: list[int], nbnd: int) -> Path:
    """Build an NSCF pw.x input from the SCF input in `scf_workdir`.

    - copies espresso.pwi → nscf_workdir/nscf.in
    - sets calculation = 'nscf'
    - injects nbnd into the &system namelist
    - strips the `K_POINTS automatic` block (header + grid line)
    - appends an explicit K_POINTS crystal block from kmesh.pl
    """
    nscf_workdir.mkdir(parents=True, exist_ok=True)
    src = scf_workdir / "espresso.pwi"
    dst = nscf_workdir / "nscf.in"

    lines = src.read_text().splitlines()
    out: list[str] = []
    skip_next = False
    in_system = False
    nbnd_injected = False
    for line in lines:
        if skip_next:
            skip_next = False
            continue
        stripped = line.strip()

        # Track when we're inside the &system namelist
        if stripped.lower().startswith("&system"):
            in_system = True
        elif in_system and stripped == "/":
            # Inject nbnd right before the closing '/'
            if not nbnd_injected:
                indent = line[: len(line) - len(line.lstrip())] or "   "
                out.append(f"{indent}nbnd             = {nbnd}")
                nbnd_injected = True
            in_system = False

        if stripped.startswith("calculation"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}calculation      = 'nscf'")
        elif stripped.startswith("outdir"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}outdir           = '{nscf_workdir / 'tmp'}'")
        elif stripped.startswith("nbnd"):
            # Replace any existing nbnd line
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}nbnd             = {nbnd}")
            nbnd_injected = True
        elif stripped.startswith("K_POINTS"):
            # drop this line and the grid-spec line that follows
            skip_next = True
            continue
        else:
            out.append(line)

    # strip any trailing blank lines, then append the kmesh block
    while out and out[-1].strip() == "":
        out.pop()

    nx, ny, nz = nscf_kgrid
    kblock = _kmesh_block(nx, ny, nz).rstrip()

    dst.write_text("\n".join(out) + "\n\n" + kblock + "\n")
    return dst


def resolve_scf_workdir(formula: str, cache_key: str) -> "tuple[Path, dict]":
    """Locate the completed parent SCF run by its run_dft cache key.

    Globs DFT_RUNS_DIR/*_{cache_key} so both naming schemes match: bulk
    `{formula}_{key}` (which pre-rename monolayer dirs also use) and
    monolayer `{formula}_2d_{key}`. Raises ValueError unless exactly one
    completed, converged run exists. Returns (workdir, parsed result.json).
    """
    matches = sorted(p for p in DFT_RUNS_DIR.glob(f"*_{cache_key}") if p.is_dir())
    if not matches:
        raise ValueError(
            f"no SCF run found for cache_key '{cache_key}' under {DFT_RUNS_DIR}; "
            "run run_dft first and pass the cache_key from its result."
        )
    complete = [p for p in matches
                if (p / "tmp").exists() and (p / "result.json").exists()
                and (p / "espresso.pwi").exists()]
    if not complete:
        names = ", ".join(p.name for p in matches)
        raise ValueError(
            f"SCF dir(s) {names} match cache_key '{cache_key}' but are incomplete "
            "(missing tmp/, result.json, or espresso.pwi); re-run run_dft."
        )
    if len(complete) > 1:
        names = ", ".join(p.name for p in complete)
        raise ValueError(
            f"multiple completed SCF dirs match cache_key '{cache_key}': {names}; "
            "remove the stale duplicate."
        )
    workdir = complete[0]
    try:
        result = json.loads((workdir / "result.json").read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"corrupt result.json at {workdir}: {e}")
    if result.get("dft_status") != "done":
        raise ValueError(
            f"parent SCF at {workdir} has dft_status="
            f"'{result.get('dft_status')}'; a converged run is required."
        )
    if result.get("formula") != formula:
        raise ValueError(
            f"cache_key '{cache_key}' belongs to formula "
            f"'{result.get('formula')}', not '{formula}'."
        )
    return workdir, result


def stage_nscf(scf_workdir: Path, nscf_workdir: Path,
               nscf_kgrid: list[int], nbnd: int) -> Path:
    """Stage an NSCF run next to the SCF: copy tmp/ (wavefunctions, charge
    density) so pw.x can pick up where SCF left off, then write nscf.in."""
    scf_workdir = Path(scf_workdir).resolve()
    nscf_workdir = Path(nscf_workdir).resolve()
    nscf_workdir.mkdir(parents=True, exist_ok=True)
    src_tmp = scf_workdir / "tmp"
    dst_tmp = nscf_workdir / "tmp"
    if not src_tmp.exists():
        raise FileNotFoundError(f"SCF tmp/ missing: {src_tmp}")
    if dst_tmp.exists():
        shutil.rmtree(dst_tmp)
    shutil.copytree(src_tmp, dst_tmp)
    return make_nscf_input(scf_workdir, nscf_workdir, nscf_kgrid, nbnd)


def run_nscf(nscf_workdir: Path) -> Path:
    """Run pw.x on nscf.in inside `nscf_workdir`. Blocks until pw.x exits.
    Writes nscf.out next to nscf.in and returns its path."""
    nscf_workdir = Path(nscf_workdir).resolve()
    inp = nscf_workdir / "nscf.in"
    out = nscf_workdir / "nscf.out"
    if not inp.exists():
        raise FileNotFoundError(f"missing NSCF input: {inp}")

    pw_x = str(Path(QE_BIN) / "pw.x")
    cmd = ["mpirun", "-n", str(MPI_NPROCS), pw_x, "-in", str(inp)]
    with out.open("w") as fout:
        subprocess.run(cmd, cwd=nscf_workdir, stdout=fout,
                       stderr=subprocess.STDOUT, check=True)

    text = out.read_text()
    if "JOB DONE" not in text:
        raise RuntimeError(f"NSCF did not finish cleanly; see {out}")
    return out


# ---- Wannier90 staging --------------------------------------------------
def _detect_prefix(nscf_workdir: Path) -> str:
    """Recover the QE `prefix` from the NSCF run by inspecting tmp/<prefix>.save.
    pw2wannier90 needs this to locate the saved wavefunctions."""
    tmp = Path(nscf_workdir) / "tmp"
    saves = list(tmp.glob("*.save"))
    if not saves:
        raise FileNotFoundError(f"no <prefix>.save directory in {tmp}")
    return saves[0].name[: -len(".save")]


def _compress_ranges(bands: list[int]) -> str:
    """Collapse a sorted band list into Wannier90 range syntax: [1,2,3,5] -> '1-3,5'."""
    bands = sorted(set(bands))
    parts: list[str] = []
    start = prev = bands[0]
    for b in bands[1:]:
        if b == prev + 1:
            prev = b
        else:
            parts.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = b
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


def _unit_cell_block(atoms) -> str:
    lines = ["begin unit_cell_cart", "ang"]
    for v in atoms.cell[:]:
        lines.append(f"{v[0]:14.8f}{v[1]:14.8f}{v[2]:14.8f}")
    lines.append("end unit_cell_cart")
    return "\n".join(lines)


def _atoms_frac_block(atoms) -> str:
    lines = ["begin atoms_frac"]
    for sym, pos in zip(atoms.get_chemical_symbols(),
                        atoms.get_scaled_positions(wrap=False)):
        lines.append(f"{sym:<3}{pos[0]:14.8f}{pos[1]:14.8f}{pos[2]:14.8f}")
    lines.append("end atoms_frac")
    return "\n".join(lines)


def _kpoints_block(nscf_kgrid: list[int]) -> str:
    """Explicit Gamma-centred uniform k-mesh for the .win, generated with the
    same nested ordering (k fastest) that kmesh.pl uses for the NSCF K_POINTS so
    pw2wannier90 can match every k-point one-to-one."""
    n1, n2, n3 = nscf_kgrid
    lines = ["begin kpoints"]
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                lines.append(f"{i / n1:14.8f}{j / n2:14.8f}{k / n3:14.8f}")
    lines.append("end kpoints")
    return "\n".join(lines)


def write_win(wann_workdir: Path, seed: str, atoms, num_bands: int,
              num_wann: int, projections: list[str], nscf_kgrid: list[int],
              exclude_bands: Optional[list[int]] = None,
              dis_win_min: Optional[float] = None,
              dis_win_max: Optional[float] = None,
              dis_froz_min: Optional[float] = None,
              dis_froz_max: Optional[float] = None,
              dis_num_iter: int = 500, num_iter: int = 2000) -> Path:
    """Write the `<seed>.win` master input from the agent's Wannierization
    decisions (projections / exclusions / disentanglement windows) plus the
    crystal structure and the NSCF k-mesh."""
    blocks: list[str] = []

    blocks.append("\n".join(
        ["begin projections"] + [f" {p}" for p in projections] + ["end projections"]
    ))
    blocks.append("guiding_centres = true")

    header = [f"num_bands = {num_bands}", f"num_wann  = {num_wann}"]
    if exclude_bands:
        header.append(f"exclude_bands : {_compress_ranges(exclude_bands)}")
    blocks.append("\n".join(header))

    # Disentanglement is only meaningful when there are more bands than WFs.
    if num_bands > num_wann:
        dis = [f"dis_num_iter = {dis_num_iter}"]
        if dis_win_min is not None:
            dis.append(f"dis_win_min  = {dis_win_min}")
        if dis_win_max is not None:
            dis.append(f"dis_win_max  = {dis_win_max}")
        if dis_froz_min is not None:
            dis.append(f"dis_froz_min = {dis_froz_min}")
        if dis_froz_max is not None:
            dis.append(f"dis_froz_max = {dis_froz_max}")
        blocks.append("\n".join(dis))

    # conv_window enables the wannierise early-stop convergence test (Wannier90's
    # default is -1 = off, which would run all num_iter cycles and never print the
    # "convergence criteria satisfied" line parse_wout looks for).
    blocks.append(f"num_iter = {num_iter}\nconv_tol = 1.0E-10\nconv_window = 5\n"
                  "iprint = 2")
    # Outputs needed downstream (e.g. Perturbo): real-space Hamiltonian, U
    # matrices, and WF centres.
    blocks.append("write_hr = .true.\nwrite_u_matrices = .true.\nwrite_xyz = .true.")
    blocks.append(f"mp_grid : {nscf_kgrid[0]} {nscf_kgrid[1]} {nscf_kgrid[2]}")
    blocks.append(_unit_cell_block(atoms))
    blocks.append(_atoms_frac_block(atoms))
    blocks.append(_kpoints_block(nscf_kgrid))

    win = wann_workdir / f"{seed}.win"
    win.write_text("\n\n".join(blocks) + "\n")
    return win


def write_pw2wan(wann_workdir: Path, nscf_tmp: Path, prefix: str, seed: str,
                 spin_component: str = "none") -> Path:
    """Write the pw2wannier90 input. `outdir` points at the NSCF tmp/ so the
    overlaps/projections are read in place; .amn/.mmn/.eig land in wann_workdir."""
    txt = (
        "&inputpp\n"
        f"  outdir = '{nscf_tmp}'\n"
        f"  prefix = '{prefix}'\n"
        f"  seedname = '{seed}'\n"
        f"  spin_component = '{spin_component}'\n"
        "  write_mmn = .true.\n"
        "  write_amn = .true.\n"
        "  write_unk = .false.\n"
        "/\n"
    )
    p = wann_workdir / "pw2wan.in"
    p.write_text(txt)
    return p


def run_w90_pp(wann_workdir: Path, seed: str) -> Path:
    """`wannier90.x -pp` (serial): reads <seed>.win, writes <seed>.nnkp listing
    the overlaps pw2wannier90 must compute."""
    subprocess.run([W90_X, "-pp", seed], cwd=wann_workdir,
                   check=True, capture_output=True, text=True)
    nnkp = wann_workdir / f"{seed}.nnkp"
    if not nnkp.exists():
        raise RuntimeError(f"wannier90 -pp produced no {nnkp}")
    return nnkp


def run_pw2wannier90(wann_workdir: Path, pw2wan_in: Path) -> Path:
    """`pw2wannier90.x` (MPI): writes <seed>.amn/.mmn/.eig into wann_workdir."""
    out = wann_workdir / "pw2wan.out"
    cmd = ["mpirun", "-n", str(MPI_NPROCS), PW2WAN_X, "-inp", str(pw2wan_in)]
    with out.open("w") as fout:
        subprocess.run(cmd, cwd=wann_workdir, stdout=fout,
                       stderr=subprocess.STDOUT, check=True)
    if "JOB DONE" not in out.read_text():
        raise RuntimeError(f"pw2wannier90 did not finish cleanly; see {out}")
    return out


def run_w90(wann_workdir: Path, seed: str) -> Path:
    """`wannier90.x <seed>` (serial): the Wannierization itself. Writes
    <seed>.wout and, per the .win flags, <seed>_hr.dat / _u.mat / _centres.xyz."""
    log = wann_workdir / f"{seed}.w90.log"
    with log.open("w") as fout:
        subprocess.run([W90_X, seed], cwd=wann_workdir, stdout=fout,
                       stderr=subprocess.STDOUT, check=True)
    wout = wann_workdir / f"{seed}.wout"
    if not wout.exists():
        raise RuntimeError(f"wannier90 produced no {wout}")
    text = wout.read_text()
    if "Exiting......" in text:
        # wannier90 reports errors in the .wout and still exits with code 0
        detail = text[text.index("Exiting......"):][:500].strip()
        raise RuntimeError(f"wannier90 failed: {detail}")
    return wout


def parse_wout(wout_path: Path) -> dict:
    """Pull the Wannierization quality metrics out of <seed>.wout: the final
    spread decomposition, convergence flags, and per-WF centres/spreads."""
    text = Path(wout_path).read_text()

    def _f(pattern: str) -> Optional[float]:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    res = {
        "omega_I": _f(r"Omega I\s+=\s+([-\d.Ee+]+)"),
        "omega_D": _f(r"Omega D\s+=\s+([-\d.Ee+]+)"),
        "omega_OD": _f(r"Omega OD\s+=\s+([-\d.Ee+]+)"),
        "omega_total_Ang2": _f(r"Omega Total\s+=\s+([-\d.Ee+]+)"),
        "dis_converged": "Disentanglement convergence criteria satisfied" in text,
        "wann_converged": "Wannierisation convergence criteria satisfied" in text,
    }

    centres: list[dict] = []
    fs = text.rfind("Final State")
    if fs != -1:
        wf_re = re.compile(
            r"WF centre and spread\s+(\d+)\s+\(\s*([-\d.]+),\s*([-\d.]+),"
            r"\s*([-\d.]+)\s*\)\s+([-\d.]+)"
        )
        for line in text[fs:].splitlines():
            m = wf_re.search(line)
            if m:
                centres.append({
                    "index": int(m.group(1)),
                    "centre_Ang": [float(m.group(2)), float(m.group(3)),
                                   float(m.group(4))],
                    "spread_Ang2": float(m.group(5)),
                })
            elif line.strip().startswith("Sum of centres"):
                break
    res["wf_centres"] = centres
    res["n_wf"] = len(centres)
    if centres:
        spreads = [c["spread_Ang2"] for c in centres]
        res["max_wf_spread_Ang2"] = max(spreads)
        res["mean_wf_spread_Ang2"] = sum(spreads) / len(spreads)
    return res


# ---- input schema --------------------------------------------------------
class Wannier90Input(BaseModel):
    formula: str = Field(..., description="Chemical formula, e.g. 'CrI3', 'MoS2'.")
    nbnd: int = Field(
        ...,
        ge=1,
        description=(
            "Number of Kohn-Sham bands for the NSCF step. This MUST be larger "
            "than the SCF default (nelec/2 for non-magnetic, nelec for magnetic). "
            "Use the band_info returned by run_dft to decide. Guidelines:\n"
            "  - Transition-metal d-block materials (Cr, Mo, W, Fe, …): include "
            "the full d-manifold plus 5-10 empty bands above for disentanglement.\n"
            "  - sp semiconductors (Si, GaAs, …): nelec/2 + 4-8 empty bands is "
            "usually sufficient.\n"
            "  - Magnetic systems (nspin=2): the SCF already reports nbnd per spin "
            "channel; add extra empty bands on top of that.\n"
            "  - When in doubt, use num_atomic_wfc from band_info as a lower "
            "bound and add ~20%% headroom."
        ),
    )
    num_wann: int = Field(
        ...,
        ge=1,
        description=(
            "Number of maximally-localized Wannier functions to build = the size "
            "of the transport-relevant manifold. This MUST equal the total number "
            "of orbitals listed in `projections` (e.g. 'Si:sp3' on 2 atoms -> 8). "
            "Pick the bands that actually carry transport near the Fermi level: "
            "for sp semiconductors the s+p valence/conduction manifold; for "
            "transition-metal compounds the d-manifold (often plus ligand-p). "
            "num_wann <= num_bands, where num_bands = nbnd - len(exclude_bands)."
        ),
    )
    projections: list[str] = Field(
        ...,
        description=(
            "Initial Wannier projections in Wannier90 syntax, one entry per line "
            "of the projections block, e.g. ['Si:sp3'], ['Cr:d'], or "
            "['Cr:d', 'I:p']. Choose the orbital CHARACTER of the target bands "
            "from chemistry/literature, not by guessing: sp3 for tetrahedral "
            "semiconductors, d for transition-metal bands at E_F, p for "
            "halide/chalcogen valence bands. The orbitals here must sum to num_wann."
        ),
    )
    exclude_bands: Optional[list[int]] = Field(
        None,
        description=(
            "1-based band indices to exclude from Wannierization (semicore / deep "
            "states far from E_F that are irrelevant for transport), e.g. [1,2,3,4]. "
            "Excluding them shrinks num_bands and keeps the disentanglement clean. "
            "Leave null if the lowest NSCF bands are already valence states you want."
        ),
    )
    dis_win_max: Optional[float] = Field(
        None,
        description=(
            "Top of the OUTER disentanglement window, in ABSOLUTE eV (same "
            "reference as band_info.fermi_energy_eV / the QE eigenvalues). REQUIRED "
            "whenever num_bands > num_wann. Set it a few eV above the highest band "
            "you want captured so there is a buffer of states to disentangle from."
        ),
    )
    dis_win_min: Optional[float] = Field(
        None,
        description=(
            "Bottom of the OUTER disentanglement window in absolute eV. Optional; "
            "if null Wannier90 uses the lowest available band. Set it to cut off "
            "low-lying bands you are not excluding outright."
        ),
    )
    dis_froz_max: Optional[float] = Field(
        None,
        description=(
            "Top of the FROZEN (inner) window in absolute eV: bands below this are "
            "reproduced exactly. Set it around/just above the Fermi level so the "
            "occupied transport bands are frozen in, typically "
            "band_info.fermi_energy_eV + ~1-2 eV. Optional but strongly recommended."
        ),
    )
    dis_froz_min: Optional[float] = Field(
        None,
        description="Bottom of the frozen window in absolute eV. Optional.",
    )
    spin_component: Literal["none", "up", "down"] = Field(
        "none",
        description=(
            "pw2wannier90 spin channel. Must match the parent SCF's nspin: "
            "'none' for non-magnetic (nspin=1). For collinear-magnetic runs "
            "(nspin=2) you MUST pick 'up' or 'down' — one channel is "
            "Wannierized per call."
        ),
    )
    nscf_kgrid: Optional[list[int]] = Field(
        default=None,
        description=(
            "k-grid for the NSCF (Wannier) step. Omit (default) to inherit the "
            "parent SCF's k-grid — the recommended choice; it keeps the qe2pert "
            "commensurability chain intact automatically. Indices along vacuum "
            "axes are auto-forced to 1."
        ),
    )
    parent_scf_cache_key: str = Field(
        ...,
        description=(
            "cache_key returned by run_dft for the parent SCF (bulk or "
            "monolayer). REQUIRED: the SCF must already exist and be "
            "converged — call run_dft first."
        ),
    )


TOOL_SPEC = {
    "name": "run_wannier90",
    "description": (
        "Run the NSCF -> Wannier90 pipeline on top of a completed run_dft SCF "
        "(bulk or monolayer) and report the Wannierization (this runs ONCE; it "
        "does not iterate to optimize the windows). REQUIRED for bulk and 2D "
        "alike: parent_scf_cache_key = the cache_key from the run_dft result. "
        "Before calling, you MUST inspect the band_info from run_dft "
        "(nelec, nbnd_scf, num_atomic_wfc, elements, fermi_energy_eV) and decide, "
        "with physical/literature reasoning: (1) nbnd for the NSCF; (2) which deep "
        "bands to drop via exclude_bands; (3) num_wann + projections capturing the "
        "ORBITAL CHARACTER of the transport bands; (4) the disentanglement windows "
        "(dis_win_max, and dis_froz_max ~ the Fermi level) in absolute eV. The result "
        "reports the final spread (Omega_I/D/OD/total), convergence flags, and per-WF "
        "centres/spreads — the quality metrics of the Wannierization."
    ),
    "input_schema": Wannier90Input.model_json_schema(),
}


def run_wannier90(formula: str, nbnd: int, num_wann: int,
                  projections: list[str], parent_scf_cache_key: str,
                  exclude_bands: Optional[list[int]] = None,
                  dis_win_max: Optional[float] = None,
                  dis_win_min: Optional[float] = None,
                  dis_froz_max: Optional[float] = None,
                  dis_froz_min: Optional[float] = None,
                  spin_component: str = "none",
                  nscf_kgrid: Optional[list[int]] = None) -> dict:
    """Run NSCF -> Wannier90 once on top of the completed parent SCF (bulk or
    monolayer, resolved by cache key) and report the Wannierization results."""
    # ---- validate the Wannierization choices up front (cheap, before any QE) --
    num_bands = nbnd - len(exclude_bands or [])
    if num_wann > num_bands:
        return {"status": "failed",
                "error": (f"num_wann ({num_wann}) exceeds num_bands ({num_bands} = "
                          f"nbnd {nbnd} - {len(exclude_bands or [])} excluded). "
                          "Lower num_wann or raise nbnd.")}
    if num_bands > num_wann and dis_win_max is None:
        return {"status": "failed",
                "error": (f"disentanglement is needed (num_bands {num_bands} > "
                          f"num_wann {num_wann}) but dis_win_max was not set. "
                          "Provide an outer-window top in absolute eV.")}

    # 1. Resolve the completed parent SCF by cache key (bulk and 2D alike)
    try:
        scf_workdir, scf_result = resolve_scf_workdir(formula, parent_scf_cache_key)
    except ValueError as e:
        return {"status": "failed", "error": str(e)}

    scf_nspin = int(scf_result.get("nspin", 1))
    if scf_nspin == 2 and spin_component == "none":
        return {"status": "failed",
                "error": (f"parent SCF (cache_key {parent_scf_cache_key}) is "
                          "spin-polarized (nspin=2); set spin_component to 'up' "
                          "or 'down' — one channel is Wannierized per call.")}
    if scf_nspin == 1 and spin_component != "none":
        return {"status": "failed",
                "error": (f"parent SCF (cache_key {parent_scf_cache_key}) is "
                          "non-magnetic (nspin=1); spin_component must be 'none'.")}

    # The structure QE actually used (bulk or carved monolayer) -- never
    # re-fetched from Materials Project, so the .win cell/positions always
    # match the NSCF wavefunctions.
    atoms = ase_read(scf_workdir / "espresso.pwi")
    label = f"{formula}_2d" if scf_result.get("monolayer") else formula

    if nscf_kgrid is None:
        # Inherit the parent SCF's grid (the post-forcing grid actually run):
        # NSCF == SCF keeps the qe2pert k/q commensurability chain automatic.
        nscf_kgrid = list(scf_result["kgrid"])
        nscf_kgrid_source = "inherited from SCF"
    else:
        nscf_kgrid = list(nscf_kgrid)
        nscf_kgrid_source = "user"
    # Never sample k along a vacuum direction (same guard as run_dft). Forced
    # before staging so nscf.in, mp_grid, and the .win kpoints all agree.
    kgrid_note = None
    vacuum_dirs = [i for i in _vacuum_axes(atoms) if nscf_kgrid[i] > 1]
    if vacuum_dirs:
        for i in vacuum_dirs:
            nscf_kgrid[i] = 1
        kgrid_note = (f"nscf_kgrid forced to 1 along vacuum axis(es) {vacuum_dirs}: "
                      "the structure has no periodic images to disperse across there")

    # 2. Stage and run NSCF (writes the wavefunctions Wannier90 needs).
    # pw.x's davidson NSCF intermittently dies in cdiaghg ("S matrix not
    # positive definite") from its random starting wavefunctions; one clean
    # re-stage + retry absorbs the flake.
    nscf_workdir = W90_RUNS_DIR / f"{label}_{parent_scf_cache_key}_nscf"
    try:
        stage_nscf(scf_workdir, nscf_workdir, nscf_kgrid, nbnd)
        run_nscf(nscf_workdir)
    except subprocess.CalledProcessError:
        stage_nscf(scf_workdir, nscf_workdir, nscf_kgrid, nbnd)
        run_nscf(nscf_workdir)

    # 3. Build the Wannier90 inputs from the agent's decisions + the structure
    seed = label
    wann_dirname = f"{label}_{parent_scf_cache_key}_wann"
    if spin_component != "none":
        wann_dirname += f"_{spin_component}"   # up/down runs must not clobber
    wann_workdir = W90_RUNS_DIR / wann_dirname
    wann_workdir.mkdir(parents=True, exist_ok=True)
    # wann dirs are reused in place; drop outputs of any previous attempt so
    # downstream tools (run_qe2pert) never mix files from different runs.
    for stale in (f"{seed}_u.mat", f"{seed}_u_dis.mat",
                  f"{seed}_centres.xyz", f"{seed}_hr.dat"):
        p = wann_workdir / stale
        if p.exists():
            p.unlink()
    win = write_win(wann_workdir, seed, atoms, num_bands, num_wann, projections,
                    nscf_kgrid, exclude_bands=exclude_bands,
                    dis_win_min=dis_win_min, dis_win_max=dis_win_max,
                    dis_froz_min=dis_froz_min, dis_froz_max=dis_froz_max)
    prefix = _detect_prefix(nscf_workdir)
    nscf_tmp = (Path(nscf_workdir) / "tmp").resolve()
    pw2wan_in = write_pw2wan(wann_workdir, nscf_tmp, prefix, seed, spin_component)

    # 4. Run wannier90 -pp -> pw2wannier90 -> wannier90
    try:
        run_w90_pp(wann_workdir, seed)
        run_pw2wannier90(wann_workdir, pw2wan_in)
        wout = run_w90(wann_workdir, seed)
    except (subprocess.CalledProcessError, RuntimeError) as e:
        werr = list(wann_workdir.glob(f"{seed}*.werr"))
        detail = (werr[0].read_text() if werr
                  else getattr(e, "stderr", None) or str(e))
        out = {"status": "wannier90_failed",
               "error": detail[:2000],
               "nscf_workdir": str(nscf_workdir),
               "wann_workdir": str(wann_workdir),
               "win_file": str(win)}
        if "fewer states than number of target WFs" in detail:
            out["hint"] = ("dis_win_max is too low: at some k-point fewer than "
                           "num_wann bands lie inside the outer window. Raise "
                           "dis_win_max by several eV (high bands disperse far "
                           "above their Gamma energies).")
        return out

    # wannier90 sometimes dies without the .wout error marker; never report
    # "done" unless the outputs downstream tools consume actually exist.
    expected = [wann_workdir / f"{seed}_u.mat",
                wann_workdir / f"{seed}_centres.xyz"]
    missing = [p.name for p in expected if not p.exists()]
    if missing:
        return {"status": "wannier90_failed",
                "error": (f"wannier90 exited without writing "
                          f"{', '.join(missing)}; see {wout}"),
                "nscf_workdir": str(nscf_workdir),
                "wann_workdir": str(wann_workdir),
                "win_file": str(win)}

    results = parse_wout(wout)
    band_info = scf_result.get("band_info")
    if band_info is None:
        try:
            band_info = parse_scf_band_info(scf_workdir)
        except Exception:
            band_info = None
    out = {
        "status": "done",
        "label": label,
        "scf_cache_key": parent_scf_cache_key,
        "scf_workdir": str(scf_workdir),
        "nscf_workdir": str(nscf_workdir),
        "wann_workdir": str(wann_workdir),
        "seed": seed,
        "win_file": str(win),
        "wout_file": str(wout),
        "hr_file": str(wann_workdir / f"{seed}_hr.dat"),
        "nbnd": nbnd,
        "num_bands": num_bands,
        "num_wann": num_wann,
        "projections": projections,
        "exclude_bands": exclude_bands,
        "spin_component": spin_component,
        "nscf_kgrid": nscf_kgrid,
        "nscf_kgrid_source": nscf_kgrid_source,
        "disentanglement_windows": {
            "dis_win_min": dis_win_min, "dis_win_max": dis_win_max,
            "dis_froz_min": dis_froz_min, "dis_froz_max": dis_froz_max,
        },
        "wannier90": results,
        "band_info": band_info,
    }
    if kgrid_note is not None:
        out["kgrid_note"] = kgrid_note
    return out
