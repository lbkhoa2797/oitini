from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from typing import Optional

from ase.io import read as ase_read

from config import PH_RUNS_DIR, QE_BIN, MPI_NPROCS
from tools.wannier90 import resolve_scf_workdir, _detect_prefix
from tools.dft import _vacuum_axes
from tools.grids import qgrid_from_kgrid

from pydantic import BaseModel, Field


# ph.x applies no acoustic sum rule, so the 3 acoustic modes at Gamma come out
# at +-a few cm-1 (often slightly negative) -- numerical noise, not instability.
ACOUSTIC_TOL_CM1 = 15.0

# tr2_ph may never be looser than this (Perturbo recommends 1e-14 for e-ph).
TR2_PH_CAP = 1.0e-14


def _fortran_d(x: float) -> str:
    """1e-14 -> '1.0d-14' (Fortran double-precision literal for namelists)."""
    return f"{x:.1e}".replace("e", "d")


def _scf_conv_thr(scf_workdir: Path) -> float:
    """conv_thr the parent SCF actually ran with, read from its espresso.pwi
    (result.json does not record it)."""
    text = (Path(scf_workdir) / "espresso.pwi").read_text()
    m = re.search(r"conv_thr\s*=\s*([\d.eEdD+-]+)", text)
    if m is None:
        raise ValueError(f"no conv_thr found in {scf_workdir}/espresso.pwi")
    return float(m.group(1).lower().replace("d", "e"))


def _derive_tr2_ph(conv_thr: float) -> float:
    """DFPT threshold: 2 orders of magnitude tighter than the SCF conv_thr,
    but never looser than TR2_PH_CAP. Heuristic guardrail -- the response
    cannot be more converged than the ground state it is built on. (For a
    very loose SCF the cap still demands 1e-14, which may converge slowly;
    run_dft's hardcoded conv_thr=1e-9 keeps that case out of the pipeline.)"""
    return min(TR2_PH_CAP, conv_thr * 1e-2)


def stage_ph(scf_workdir: Path, ph_workdir: Path,
               ph_qgrid: list[int], tr2_ph: float, is_metal,
               recover_mode=False) -> Path:
    """Stage a PH run following the SCF: copy 'tmp/' dir so ph.x can interpret ph.in"""
    scf_workdir = Path(scf_workdir).resolve()
    ph_workdir = Path(ph_workdir).resolve()
    ph_workdir.mkdir(parents=True, exist_ok=True)

    src_tmp = scf_workdir / "tmp"
    dst_tmp = ph_workdir / "tmp"

    if not recover_mode:
        if dst_tmp.exists():
            shutil.rmtree(dst_tmp)
        shutil.copytree(src_tmp, dst_tmp)
        # also clear stale dyn files from any previous attempt,
        # so collect never mixes runs:
        for f in ph_workdir.glob("*.dyn*"):
            f.unlink()
    
    dst = ph_workdir / "ph.in"

    nx, ny, nz = ph_qgrid
    prefix = _detect_prefix(scf_workdir)

    epsil = f"false" if is_metal else f"true"
    recover = f"true" if recover_mode else f"false" 

    content = f"""phonons on uniform grid
&inputph
   tr2_ph   = {_fortran_d(tr2_ph)},
   prefix   = '{prefix}'
   outdir   = 'tmp'
   fildyn   = '{prefix}.dyn.xml'
   fildvscf = 'dvscf'
   ldisp    = .true.
   epsil    = .{epsil}.
   recover  = .{recover}.
   nq1 = {nx}, nq2 = {ny}, nq3 = {nz}
/
"""
    dst.write_text(content)

    return dst

def run_ph(ph_workdir: Path) -> Path:
    """Run ph.x on ph.in inside `ph_workdir`. Blocks until ph.x exits.
    Writes ph.out next to ph.in and returns its path."""
    ph_workdir = Path(ph_workdir).resolve()
    inp = ph_workdir / "ph.in"
    out = ph_workdir / "ph.out"
    if not inp.exists():
        raise FileNotFoundError(f"missing ph input: {inp}")

    ph_x = str(Path(QE_BIN) / "ph.x")
    cmd = ["mpirun", "-n", str(MPI_NPROCS), ph_x, "-in", str(inp)]
    with out.open("w") as fout:
        subprocess.run(cmd, cwd=ph_workdir, stdout=fout,
                       stderr=subprocess.STDOUT, check=True)

    text = out.read_text()

    if "JOB DONE" not in text:
        raise RuntimeError(f"PH did not finish cleanly; see {out}")
    return out


def parse_dyn0(ph_workdir: Path, prefix: str) -> int:
    """Number of irreducible q-points, read from {prefix}.dyn0 (2nd line).
    This is the only place the q count is known -- never hardcode it."""
    path = Path(ph_workdir) / f"{prefix}.dyn0"
    if not path.exists():
        raise FileNotFoundError(f"missing q-list file {path}; "
                                "ph.x likely died before the q-point setup")
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    n_q = int(lines[1])
    if len(lines) - 2 != n_q:
        raise ValueError(f"{path} claims {n_q} q-points but lists {len(lines) - 2}")
    return n_q


def parse_dyn_xml(dyn_path: Path) -> dict:
    """Frequencies + q-vector from one {prefix}.dynN.xml dynamical-matrix file.

    The file stores the whole star of the irreducible q (one DYNAMICAL_MAT_.n
    block per star member) but a single frequency block, so the first Q_POINT
    is taken as representative. Each OMEGA.n element holds two floats:
    frequency in THz then cm-1. Unlike ph.out, these files exist for every q
    even after a recovery run, so they are the recovery-proof source of truth."""
    dyn_path = Path(dyn_path)
    root = ET.parse(dyn_path).getroot()

    q_el = root.find(".//Q_POINT")
    if q_el is None or q_el.text is None:
        raise ValueError(f"no Q_POINT element in {dyn_path}")
    q = [float(x) for x in q_el.text.split()]

    freq_block = root.find("FREQUENCIES_THZ_CMM1")
    if freq_block is None:
        raise ValueError(f"no FREQUENCIES_THZ_CMM1 block in {dyn_path}")

    omegas = sorted(
        (el for el in freq_block if el.tag.startswith("OMEGA.")),
        key=lambda el: int(el.tag.split(".")[1]),
    )
    freqs_THz, freqs_cm1 = [], []
    for el in omegas:
        thz, cm1 = (float(x) for x in (el.text or "").split())
        freqs_THz.append(thz)
        freqs_cm1.append(cm1)

    return {"q": q, "freqs_THz": freqs_THz, "freqs_cm1": freqs_cm1}


def classify_modes(per_q: list[dict],
                   acoustic_tol_cm1: float = ACOUSTIC_TOL_CM1) -> dict:
    """Soft-mode check over parse_dyn_xml results for all irreducible q.

    Imaginary frequencies are reported by ph.x as negative numbers. At Gamma
    the 3 lowest-|freq| modes are exempt within +-acoustic_tol_cm1 (missing
    acoustic sum rule); any other negative frequency is a real instability."""
    imaginary_modes = []
    non_exempt_freqs = []
    gamma_freqs_THz = None

    for block in per_q:
        q = block["q"]
        cm1 = block["freqs_cm1"]
        exempt = set()
        if math.sqrt(sum(x * x for x in q)) < 1e-6:
            gamma_freqs_THz = block["freqs_THz"]
            by_abs = sorted(range(len(cm1)), key=lambda i: abs(cm1[i]))
            exempt = {i for i in by_abs[:3] if abs(cm1[i]) < acoustic_tol_cm1}

        for i, f in enumerate(cm1):
            if i in exempt:
                continue
            non_exempt_freqs.append(f)
            if f < 0.0:
                imaginary_modes.append({"q": q, "mode": i + 1, "freq_cm1": f})

    return {
        "has_imaginary_modes": bool(imaginary_modes),
        "imaginary_modes": imaginary_modes,
        "min_freq_cm1": min(non_exempt_freqs) if non_exempt_freqs else None,
        "gamma_freqs_THz": gamma_freqs_THz,
        "acoustic_tol_cm1": acoustic_tol_cm1,
    }


def collect_save(ph_workdir: Path, prefix: str, n_q: int) -> dict:
    """Collect ph.x outputs into save/ for qe2pert.x (port of ph-collect.sh).

    Layout consumed by qe2pert: {prefix}.phsave/, all dyn files, and one
    dvscf per irreducible q -- q=1 lives flat in tmp/_ph0/, q>=2 in per-q
    subdirs. Every expected source is checked before anything is copied, so
    an incomplete ph.x run fails loudly instead of yielding a corrupt save/."""
    ph_workdir = Path(ph_workdir).resolve()
    ph0 = ph_workdir / "tmp" / "_ph0"

    phsave_src = ph0 / f"{prefix}.phsave"
    dvscf_srcs = {1: ph0 / f"{prefix}.dvscf1"}
    for iq in range(2, n_q + 1):
        dvscf_srcs[iq] = ph0 / f"{prefix}.q_{iq}" / f"{prefix}.dvscf1"

    dyn_files = sorted(ph_workdir.glob(f"{prefix}.dyn*"))
    expected_dyn = {f"{prefix}.dyn0"} | {f"{prefix}.dyn{iq}.xml"
                                         for iq in range(1, n_q + 1)}
    missing = [str(p) for p in (phsave_src, *dvscf_srcs.values())
               if not p.exists()]
    missing += sorted(expected_dyn - {f.name for f in dyn_files})
    if missing:
        raise FileNotFoundError(
            f"ph.x outputs incomplete, refusing to build save/ "
            f"(rerun with recover): missing {missing}")

    save_dir = ph_workdir / "save"
    if save_dir.exists():
        shutil.rmtree(save_dir)  # never mix collects from different runs
    save_dir.mkdir()

    shutil.copytree(phsave_src, save_dir / f"{prefix}.phsave")
    for f in dyn_files:
        shutil.copy2(f, save_dir / f.name)
    for iq, src in dvscf_srcs.items():
        shutil.copy2(src, save_dir / f"{prefix}.dvscf_q{iq}")

    n_dvscf = len(list(save_dir.glob(f"{prefix}.dvscf_q*")))
    if n_dvscf != n_q:
        raise RuntimeError(f"expected {n_q} dvscf files in {save_dir}, "
                           f"found {n_dvscf}")

    # dvscf files are dense complex grids: trivial for Si, GBs per q for
    # large cells -- and collect duplicates every one of them.
    size_mb = sum(f.stat().st_size for f in save_dir.rglob("*")
                  if f.is_file()) / 1e6
    return {"save_dir": str(save_dir), "save_size_MB": round(size_mb, 1)}


# ---- input schema --------------------------------------------------------
class PhononInput(BaseModel):
    formula: str = Field(..., description="Chemical formula, e.g. 'CrI3', 'MoS2'.")
    parent_scf_cache_key: str = Field(
        ...,
        description=(
            "cache_key returned by run_dft for the parent SCF. "
            "REQUIRED: the SCF must already exist and be "
            "converged — run_dft must be run first."
        ),
    )

    ph_qgrid: Optional[list[int]] = Field(
        default=None,
        description=(
            "q-grid for the Phonon (ph.x) step. Omit (default) to derive it "
            "from the parent SCF k-grid — per direction, half of an even count "
            "above 4 points, else the full count — always an "
            "integer divisor, so commensurability holds by construction. Each "
            "explicit dimension must be an integer factor of the SCF and NSCF "
            "k-grids. Indices along vacuum axes are auto-forced to 1."
        ),
    )


TOOL_SPEC = {
    "name": "create_ph_save",
    "description": (
        "Run Quantum ESPRESSO's ph.x (DFPT phonons) over ALL irreducible "
        "q-points of ph_qgrid on top of a completed run_dft SCF, check "
        "dynamical stability, and collect the save/ folder that the future "
        "qe2pert.x (electron-phonon) step consumes. REQUIRED: "
        "parent_scf_cache_key = the cache_key from the run_dft result (the "
        "SCF must be converged — call run_dft first). ph_qgrid may be "
        "omitted: the default derives from the SCF k-grid (half of even "
        "counts above 4, else the full count) and is "
        "commensurate by construction. Commensurability rule for explicit "
        "grids: each SCF k-grid dimension — and the nscf_kgrid you later "
        "pass to run_wannier90 — must be an integer multiple of the matching "
        "ph_qgrid dimension. Physics knobs are set automatically: epsil "
        "(Born charges + dielectric tensor at Gamma) is on for insulators "
        "and off for metals per the SCF's is_metal; the DFPT threshold "
        "tr2_ph follows the SCF conv_thr (capped at 1e-14, the Perturbo "
        "recommendation). RUNTIME: 10-100x the SCF — minutes on test grids "
        "([2,2,2]), hours on production grids ([4,4,4]+); the call blocks "
        "until done. Results are cached: an identical re-call returns "
        "instantly with ph_status='cached', and an interrupted or failed "
        "run auto-resumes from its checkpoints on the next identical call, "
        "so on failure simply retry. The result reports the stability "
        "verdict — has_imaginary_modes, imaginary_modes, min_freq_cm1, "
        "gamma_freqs_THz (quote these when judging whether the structure "
        "is dynamically stable; a true imaginary mode means the structure "
        "is unstable and e-ph results would be meaningless) — plus "
        "n_qpoints_irr and save_dir/save_size_MB, the phonon half of the "
        "qe2pert input. The Wannier90 half comes from run_wannier90, which "
        "can run off the same SCF in parallel with this tool."
    ),
    "input_schema": PhononInput.model_json_schema(),
}


def create_ph_save(formula: str, parent_scf_cache_key: str,
                  ph_qgrid: Optional[list[int]] = None) -> dict:
    """Run ph.x once on top of the completed parent SCF and
    collect to 'save' databse for subsequent QE2PERT """

    # Resolve the completed parent SCF by cache key
    try:
        scf_workdir, scf_result = resolve_scf_workdir(formula, parent_scf_cache_key)
    except ValueError as e:
        return {"status": "failed", "error": str(e)}

    scf_kgrid = scf_result["kgrid"]
    is_metal = scf_result["is_metal"]

    if ph_qgrid is None:
        ph_qgrid = qgrid_from_kgrid(scf_kgrid)
        qgrid_source = "derived from SCF k-grid"
    else:
        ph_qgrid = list(ph_qgrid)  # never mutate the caller's list
        qgrid_source = "user"

    # Commensurability check
    for i in range(3):
        if scf_kgrid[i] % ph_qgrid[i] != 0:
            return {"status": "failed",
                    "error": f"scf_kgrid[{i}]={scf_kgrid[i]} is not an integer "
                             f"multiple of ph_qgrid[{i}]={ph_qgrid[i]}"}

    # Skip spin guard to account for when an insulator are computed with 
    # smearing for better convergence

    atoms = ase_read(scf_workdir / "espresso.pwi")
    # qgrid_note = None
    vacuum_dirs = [i for i in _vacuum_axes(atoms) if ph_qgrid[i] > 1]
    if vacuum_dirs:
        for i in vacuum_dirs:
            ph_qgrid[i] = 1

    q1, q2, q3 = ph_qgrid

    # DFPT threshold follows the parent SCF's conv_thr (capped at 1e-14)
    tr2_ph = _derive_tr2_ph(_scf_conv_thr(scf_workdir))

    # Stage and run ph.x
    label = f"{formula}_2d" if scf_result.get("monolayer") else formula

    payload = [parent_scf_cache_key, ph_qgrid, tr2_ph]
    ph_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
    ph_workdir = PH_RUNS_DIR / f"{label}_{parent_scf_cache_key}_q{q1}{q2}{q3}_{ph_key}_ph"

    cache_file = ph_workdir / "result.json"
    if cache_file.exists():
        return {**json.loads(cache_file.read_text()), "ph_status": "cached"}

    ph_out = ph_workdir / "ph.out"
    old_exists = (ph_workdir / "tmp" / "_ph0").exists()
    try:
        stage_ph(scf_workdir, ph_workdir, ph_qgrid, tr2_ph, is_metal,
                 recover_mode=old_exists)
        run_ph(ph_workdir)

        prefix = _detect_prefix(scf_workdir)
        n_q = parse_dyn0(ph_workdir, prefix)
        per_q = [parse_dyn_xml(ph_workdir / f"{prefix}.dyn{iq}.xml")
                 for iq in range(1, n_q + 1)]
        verdict = classify_modes(per_q)
        collect = collect_save(ph_workdir, prefix, n_q)
    except (subprocess.CalledProcessError, RuntimeError,
            FileNotFoundError, ValueError) as e:
        tail = ph_out.read_text()[-2000:] if ph_out.exists() else ""
        return {"status": "failed", "error": str(e), "ph_out_tail": tail,
                "ph_workdir": str(ph_workdir),
                "hint": "re-calling with the same parameters will "
                        "auto-recover from the checkpoint"}
    result = {
        "status": "done",
        "label": label,
        "prefix": prefix,
        "scf_cache_key": parent_scf_cache_key,
        "scf_workdir": str(scf_workdir),
        "scf_kgrid": scf_kgrid,
        "ph_workdir": str(ph_workdir),
        "ph_out": str(ph_out),
        "ph_qgrid": ph_qgrid,
        "qgrid_source": qgrid_source,
        "tr2_ph": tr2_ph,
        "epsil": not is_metal,
        "n_qpoints_irr": n_q,
        **verdict,   # has_imaginary_modes, imaginary_modes, min_freq_cm1,
                     # gamma_freqs_THz, acoustic_tol_cm1
        **collect,   # save_dir, save_size_MB
        "commensurability_note": (
            "the nscf_kgrid passed to run_wannier90 must also be an integer "
            "multiple of ph_qgrid in every dimension (qe2pert requirement)"),
    }
    if vacuum_dirs:
        result["qgrid_note"] = (f"ph_qgrid forced to 1 along vacuum "
                                f"axes {vacuum_dirs}")

    cache_file.write_text(json.dumps(result, indent=2, default=str))
    return result

