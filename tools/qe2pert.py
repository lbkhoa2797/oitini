from pathlib import Path
import json
import re
import subprocess
from typing import Literal, Optional

from config import PERTURBO_BIN, QE2PERT_RUNS_DIR, W90_RUNS_DIR, PH_RUNS_DIR, MPI_NPROCS
from tools.wannier90 import resolve_scf_workdir, _detect_prefix

from pydantic import BaseModel, Field


QE2PERT_X = str(Path(PERTURBO_BIN) / "qe2pert.x")

# qe2pert.x never prints QE's "JOB DONE"; environment_stop stamps this instead.
COMPLETION_MARKER = "Program was terminated on"


# ---- parent-run resolution ------------------------------------------------
def _expand_ranges(spec: str) -> list[int]:
    """Inverse of wannier90's _compress_ranges: '1-3,5' -> [1, 2, 3, 5]."""
    bands: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            bands.extend(range(int(lo), int(hi) + 1))
        else:
            bands.append(int(part))
    return sorted(set(bands))


def parse_win_params(win_path: Path) -> dict:
    """Read the Wannierization parameters qe2pert must stay consistent with
    from the .win written by write_win (run_wannier90 stores no result.json)."""
    text = win_path.read_text()

    def _int(pattern: str, name: str) -> int:
        m = re.search(pattern, text)
        if m is None:
            raise ValueError(f"no {name} found in {win_path}")
        return int(m.group(1))

    params = {
        "num_bands": _int(r"num_bands\s*=\s*(\d+)", "num_bands"),
        "num_wann": _int(r"num_wann\s*=\s*(\d+)", "num_wann"),
    }

    m = re.search(r"mp_grid\s*:\s*(\d+)\s+(\d+)\s+(\d+)", text)
    if m is None:
        raise ValueError(f"no mp_grid found in {win_path}")
    params["nk_grid"] = [int(g) for g in m.groups()]

    m = re.search(r"exclude_bands\s*:\s*([\d,\-]+)", text)
    params["exclude_bands"] = _expand_ranges(m.group(1)) if m else []

    m = re.search(r"dis_win_min\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)", text)
    params["dis_win_min"] = float(m.group(1)) if m else None
    return params


def derive_band_window(num_bands: int, exclude_bands: list[int]) -> "tuple[int, int]":
    """Map the Wannierization band set onto qe2pert's contiguous DFT band
    range: dft_band_min/max = first/last NSCF band kept after exclusions.
    (qe2pert reads a contiguous range; non-contiguous exclusions are not
    specially handled.)"""
    nbnd = num_bands + len(exclude_bands)
    kept = sorted(set(range(1, nbnd + 1)) - set(exclude_bands))
    if not kept:
        raise ValueError("exclude_bands leaves no NSCF bands for qe2pert")
    return kept[0], kept[-1]


def resolve_wannier_run(label: str, cache_key: str,
                        spin_component: str) -> "tuple[Path, Path, str, dict]":
    """Locate the completed NSCF + Wannier90 runs for this SCF cache key and
    parse the .win parameters. Raises ValueError with run-the-prerequisite
    guidance so the agent knows what to do next."""
    hint = (f"run run_wannier90 with parent_scf_cache_key='{cache_key}'"
            + (f" and spin_component='{spin_component}'"
               if spin_component != "none" else "")
            + " first")

    nscf_workdir = W90_RUNS_DIR / f"{label}_{cache_key}_nscf"
    if not nscf_workdir.is_dir():
        raise ValueError(f"no NSCF run at {nscf_workdir}; {hint}.")
    try:
        prefix = _detect_prefix(nscf_workdir)
    except FileNotFoundError:
        raise ValueError(
            f"NSCF run at {nscf_workdir} has no tmp/<prefix>.save; {hint}.")
    nscf_out = nscf_workdir / "nscf.out"
    if not nscf_out.exists() or "JOB DONE" not in nscf_out.read_text():
        raise ValueError(f"NSCF at {nscf_workdir} did not finish cleanly; {hint}.")

    seed = label
    wann_dirname = f"{label}_{cache_key}_wann"
    if spin_component != "none":
        wann_dirname += f"_{spin_component}"   # matches run_wannier90's naming
    wann_workdir = W90_RUNS_DIR / wann_dirname
    if not wann_workdir.is_dir():
        raise ValueError(f"no Wannier90 run at {wann_workdir}; {hint}.")
    win = wann_workdir / f"{seed}.win"
    if not win.exists():
        raise ValueError(f"missing {win}; {hint}.")
    win_params = parse_win_params(win)

    needed = [wann_workdir / f"{seed}.wout",
              wann_workdir / f"{seed}_u.mat",
              wann_workdir / f"{seed}_centres.xyz"]
    if win_params["num_bands"] > win_params["num_wann"]:
        needed.append(wann_workdir / f"{seed}_u_dis.mat")
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise ValueError(f"Wannier90 run at {wann_workdir} is missing "
                         f"{', '.join(missing)}; {hint}.")
    return nscf_workdir, wann_workdir, prefix, win_params


def resolve_ph_run(label: str, cache_key: str,
                   ph_qgrid: "Optional[list[int]]") -> "tuple[Path, dict]":
    """Locate the completed create_ph_save run (and its collected save/) for
    this SCF cache key, optionally disambiguated by ph_qgrid."""
    candidates = []
    for d in sorted(PH_RUNS_DIR.glob(f"{label}_{cache_key}_q*_ph")):
        rj = d / "result.json"
        if not rj.exists():
            continue
        try:
            res = json.loads(rj.read_text())
        except json.JSONDecodeError:
            continue
        if res.get("status") == "done":
            candidates.append((d, res))
    if ph_qgrid is not None:
        candidates = [(d, r) for d, r in candidates
                      if r.get("ph_qgrid") == list(ph_qgrid)]
    if not candidates:
        raise ValueError(
            f"no completed phonon run for cache_key '{cache_key}'"
            + (f" with ph_qgrid={list(ph_qgrid)}" if ph_qgrid else "")
            + f" under {PH_RUNS_DIR}; run create_ph_save with "
              f"parent_scf_cache_key='{cache_key}' first.")
    if len(candidates) > 1:
        listing = "; ".join(f"{d.name} (ph_qgrid={r.get('ph_qgrid')})"
                            for d, r in candidates)
        raise ValueError(
            f"multiple completed phonon runs match cache_key '{cache_key}': "
            f"{listing}. Pass ph_qgrid to pick one.")
    ph_workdir, ph_result = candidates[0]
    save_dir = Path(ph_result["save_dir"])
    prefix = ph_result["prefix"]
    if not (save_dir / f"{prefix}.phsave").exists():
        raise ValueError(f"phonon save/ at {save_dir} is missing "
                         f"{prefix}.phsave; re-run create_ph_save to rebuild it.")
    return ph_workdir, ph_result


# ---- staging --------------------------------------------------------------
def _relink(link: Path, target: Path) -> None:
    """Create/refresh a symlink idempotently, verifying the target exists."""
    target = Path(target).resolve()
    if not target.exists():
        raise FileNotFoundError(f"symlink target missing: {target}")
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def stage_qe2pert(workdir: Path, nscf_workdir: Path, wann_workdir: Path,
                  prefix: str, seed: str, link_u_dis: bool) -> None:
    """Stage the qe2pert workdir following the reference run.sh: a real tmp/
    holding a symlink to the NSCF <prefix>.save, plus symlinks to the
    Wannier90 rotation matrices and centres in the workdir root. Symlinks —
    not copies — so nothing is duplicated; qe2pert only reads through them
    (its writes go to tmp/ itself and the workdir). Also clears outputs left
    by a previous failed attempt."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    tmp = workdir / "tmp"
    tmp.mkdir(exist_ok=True)   # real dir: qe2pert writes {prefix}_elph.h5 here

    _relink(tmp / f"{prefix}.save", Path(nscf_workdir) / "tmp" / f"{prefix}.save")
    _relink(workdir / f"{prefix}_u.mat", Path(wann_workdir) / f"{seed}_u.mat")
    _relink(workdir / f"{prefix}_centres.xyz",
            Path(wann_workdir) / f"{seed}_centres.xyz")
    u_dis_link = workdir / f"{prefix}_u_dis.mat"
    if link_u_dis:
        _relink(u_dis_link, Path(wann_workdir) / f"{seed}_u_dis.mat")
    elif u_dis_link.is_symlink() or u_dis_link.exists():
        u_dis_link.unlink()   # stale link from a previous disentangled run

    for stale in (workdir / f"{prefix}_epr.h5", workdir / "qe2pert.out",
                  workdir / "qe2pert_output.yml", tmp / f"{prefix}_elph.h5"):
        if stale.exists():
            stale.unlink()


def write_qe2pert_in(workdir: Path, prefix: str, phdir: str,
                     nk_grid: list[int], dft_band_min: int, dft_band_max: int,
                     num_wann: int, dis_win_min: "Optional[float]" = None,
                     spin_component: str = "none",
                     system_2d: bool = False) -> Path:
    """Write qe2pert.in. asr and lwannier are deliberately left at qe2pert's
    defaults ('crystal' / .true.)."""
    nk1, nk2, nk3 = nk_grid
    lines = [
        "&qe2pert",
        f"  prefix='{prefix}'",
        "  outdir='./tmp'",
        f"  phdir='{phdir}'",
        f"  nk1={nk1}, nk2={nk2}, nk3={nk3}",
        f"  dft_band_min = {dft_band_min}",
        f"  dft_band_max = {dft_band_max}",
        f"  num_wann = {num_wann}",
        "  load_ephmat = .false.",
    ]
    if dis_win_min is not None:
        # forwarded from the .win (absolute eV) so qe2pert reconstructs the
        # same disentanglement window Wannier90 used
        lines.append(f"  dis_win_min = {dis_win_min}")
    if spin_component != "none":
        lines.append(f"  spin_component = '{spin_component}'")
    if system_2d:
        lines.append("  system_2d = .true.")
    lines.append("/")
    dst = Path(workdir) / "qe2pert.in"
    dst.write_text("\n".join(lines) + "\n")
    return dst


def run_qe2pert_x(workdir: Path) -> Path:
    """Run qe2pert.x on qe2pert.in inside `workdir`. Blocks until it exits.
    qe2pert parallelizes over k-pools only, so -npools must equal the MPI
    rank count. Writes qe2pert.out and returns its path."""
    workdir = Path(workdir).resolve()
    inp = workdir / "qe2pert.in"
    out = workdir / "qe2pert.out"
    if not inp.exists():
        raise FileNotFoundError(f"missing qe2pert input: {inp}")

    cmd = ["mpirun", "-n", str(MPI_NPROCS), QE2PERT_X,
           "-npools", str(MPI_NPROCS), "-inp", str(inp)]
    with out.open("w") as fout:
        subprocess.run(cmd, cwd=workdir, stdout=fout,
                       stderr=subprocess.STDOUT, check=True)

    if COMPLETION_MARKER not in out.read_text():
        raise RuntimeError(f"qe2pert did not finish cleanly; see {out}")
    return out


# ---- input schema ---------------------------------------------------------
class Qe2pertInput(BaseModel):
    formula: str = Field(..., description="Chemical formula, e.g. 'Si', 'CrI3'.")
    parent_scf_cache_key: str = Field(
        ...,
        description=(
            "cache_key returned by run_dft for the parent SCF. REQUIRED: "
            "BOTH run_wannier90 AND create_ph_save must already have "
            "completed for this same cache_key — this tool only combines "
            "their outputs."
        ),
    )
    ph_qgrid: Optional[list[int]] = Field(
        default=None,
        description=(
            "Only needed when several create_ph_save runs exist for this "
            "SCF: the q-grid of the one to use, e.g. [4, 4, 4]. Leave null "
            "when there is exactly one phonon run."
        ),
    )
    spin_component: Literal["none", "up", "down"] = Field(
        default="none",
        description=(
            "For spin-polarized (nspin=2) parents call this tool once per "
            "channel with 'up'/'down', matching the run_wannier90 channels; "
            "keep 'none' for non-magnetic parents."
        ),
    )


TOOL_SPEC = {
    "name": "run_qe2pert",
    "description": (
        "Run Perturbo's qe2pert.x to combine the Wannier90 outputs "
        "(run_wannier90) and the phonon save/ folder (create_ph_save) into "
        "{prefix}_epr.h5 — the electron-phonon database that perturbo.x "
        "consumes. This is the final step of the e-ph pipeline. REQUIRED: "
        "BOTH run_wannier90 AND create_ph_save must already be complete for "
        "the SAME parent_scf_cache_key; the tool locates both on disk and "
        "its error message says which prerequisite is missing. Everything "
        "else is derived automatically for consistency: the k-grid (nk1-3) "
        "from the NSCF mp_grid, dft_band_min/max from nbnd and "
        "exclude_bands, num_wann and dis_win_min from the .win file, and "
        "system_2d for monolayer parents. Pass ph_qgrid only if several "
        "phonon runs exist for the SCF (the error will list them). For "
        "spin-polarized parents call once per channel with "
        "spin_component='up'/'down'. Results are cached: an identical "
        "re-call returns instantly with qe2pert_status='cached'; after a "
        "transient failure simply re-call with the same parameters. Report "
        "epr_file and epr_size_MB in the final answer — the epr file is the "
        "deliverable the user feeds to perturbo.x."
    ),
    "input_schema": Qe2pertInput.model_json_schema(),
}


def run_qe2pert(formula: str, parent_scf_cache_key: str,
                ph_qgrid: "Optional[list[int]]" = None,
                spin_component: str = "none") -> dict:
    """Run qe2pert.x on top of completed run_wannier90 + create_ph_save runs
    and produce the {prefix}_epr.h5 electron-phonon database for Perturbo."""

    # Resolve the completed parent SCF by cache key
    try:
        scf_workdir, scf_result = resolve_scf_workdir(formula, parent_scf_cache_key)
    except ValueError as e:
        return {"status": "failed", "error": str(e)}

    # One qe2pert run per spin channel (qe2pert itself hard-errors otherwise)
    nspin = scf_result.get("nspin", 1)
    if nspin == 2 and spin_component == "none":
        return {"status": "failed",
                "error": ("parent SCF is spin-polarized (nspin=2): call "
                          "run_qe2pert once per channel with "
                          "spin_component='up' and 'down', matching the "
                          "run_wannier90 channels")}
    if nspin != 2 and spin_component != "none":
        return {"status": "failed",
                "error": (f"spin_component='{spin_component}' but the parent "
                          "SCF is not spin-polarized; use 'none'")}

    label = f"{formula}_2d" if scf_result.get("monolayer") else formula
    system_2d = bool(scf_result.get("monolayer"))

    # Resolve both prerequisite runs; a missing one is the agent's cue to go
    # run that step first.
    try:
        nscf_workdir, wann_workdir, prefix, win = resolve_wannier_run(
            label, parent_scf_cache_key, spin_component)
        ph_workdir, ph_result = resolve_ph_run(label, parent_scf_cache_key,
                                               ph_qgrid)
    except ValueError as e:
        return {"status": "failed", "error": str(e)}

    dft_band_min, dft_band_max = derive_band_window(win["num_bands"],
                                                    win["exclude_bands"])
    nk_grid = win["nk_grid"]
    phdir = str(Path(ph_result["save_dir"]).resolve())

    # qe2pert requires the electron k-grid to be commensurate with the q-grid
    for i in range(3):
        if nk_grid[i] % ph_result["ph_qgrid"][i] != 0:
            return {"status": "failed",
                    "error": (f"nscf k-grid {nk_grid} is not an integer "
                              f"multiple of ph_qgrid {ph_result['ph_qgrid']} "
                              f"in dimension {i} (qe2pert requirement); "
                              "re-run run_wannier90 or create_ph_save with "
                              "commensurate grids")}

    # One qe2pert dir per phonon run (its name already encodes scf key + qgrid)
    workdir_name = ph_workdir.name[: -len("_ph")] + "_qe2pert"
    if spin_component != "none":
        workdir_name += f"_{spin_component}"
    workdir = QE2PERT_RUNS_DIR / workdir_name

    derived = {
        "nk_grid": nk_grid,
        "dft_band_min": dft_band_min,
        "dft_band_max": dft_band_max,
        "num_wann": win["num_wann"],
        "dis_win_min": win["dis_win_min"],
        "phdir": phdir,
    }

    # Cache hit only if the epr file still exists AND the parameters derived
    # from the (possibly re-run) wannier/ph parents match the cached run —
    # wann dirs are overwritten in place, so a re-Wannierization must
    # invalidate this cache.
    cache_file = workdir / "result.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        epr_ok = cached.get("epr_file") and Path(cached["epr_file"]).exists()
        params_ok = all(cached.get(k) == v for k, v in derived.items())
        if epr_ok and params_ok:
            return {**cached, "qe2pert_status": "cached"}

    qe2pert_out = workdir / "qe2pert.out"
    try:
        stage_qe2pert(workdir, nscf_workdir, wann_workdir, prefix, seed=label,
                      link_u_dis=win["num_bands"] > win["num_wann"])
        qe2pert_in = write_qe2pert_in(
            workdir, prefix, phdir, nk_grid, dft_band_min, dft_band_max,
            win["num_wann"], dis_win_min=win["dis_win_min"],
            spin_component=spin_component, system_2d=system_2d)
        run_qe2pert_x(workdir)

        epr = workdir / f"{prefix}_epr.h5"
        if not epr.exists() or epr.stat().st_size == 0:
            raise RuntimeError(f"qe2pert finished but {epr.name} is missing "
                               "or empty")
    except (subprocess.CalledProcessError, RuntimeError,
            FileNotFoundError, ValueError) as e:
        tail = qe2pert_out.read_text()[-2000:] if qe2pert_out.exists() else ""
        return {"status": "failed", "error": str(e),
                "qe2pert_out_tail": tail, "workdir": str(workdir),
                "hint": "re-calling with the same parameters restages "
                        "and reruns"}

    result = {
        "status": "done",
        "label": label,
        "prefix": prefix,
        "scf_cache_key": parent_scf_cache_key,
        "workdir": str(workdir),
        "epr_file": str(epr.resolve()),
        "epr_size_MB": round(epr.stat().st_size / 1e6, 1),
        **derived,
        "spin_component": spin_component,
        "nscf_workdir": str(nscf_workdir),
        "wann_workdir": str(wann_workdir),
        "ph_workdir": str(ph_workdir),
        "ph_qgrid": ph_result["ph_qgrid"],
        "qe2pert_in": str(qe2pert_in),
        "qe2pert_out": str(qe2pert_out),
        "next_step_note": (f"{epr.name} is the electron-phonon database "
                           "consumed by perturbo.x"),
    }
    if system_2d:
        result["system_2d"] = True
    if ph_result.get("has_imaginary_modes"):
        result["stability_warning"] = (
            "the parent phonon run reported imaginary modes — e-ph results "
            "built on an unstable structure are unreliable")
    cache_file.write_text(json.dumps(result, indent=2, default=str))
    return result
