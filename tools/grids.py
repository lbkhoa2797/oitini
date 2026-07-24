"""Adaptive k/q-grid defaults derived from the reciprocal lattice.

k-grid convention: Materials Cloud / aiida k-point distance with the 2*pi
factor included, n_i = max(1, ceil(|b_i| / kdist)) where
|b_i| = 2*pi * |row_i of cell.reciprocal()| (ASE's Cell.reciprocal() omits
the 2*pi). DEFAULT_KPOINTS_DISTANCE below is the single knob (Materials
Cloud protocols: 0.30 coarse/screening, 0.15 medium, 0.10 fine).

q-grid convention: half the k-grid per direction for even counts above 4;
counts of 4 or fewer -- and odd counts, which only arise from user-passed
grids -- keep the full k value. The result always divides the k-grid, so the
qe2pert commensurability requirement holds by construction.
"""
import math

import numpy as np

DEFAULT_KPOINTS_DISTANCE = 0.30   # 1/Angstrom (Materials Cloud "coarse" -- fast screening)


def kgrid_from_spacing(atoms, kdist: float = DEFAULT_KPOINTS_DISTANCE) -> list[int]:
    """Monkhorst-Pack grid for *atoms* at a target k-point spacing (1/A).

    Counts are rounded up to even (1 stays 1) so qgrid_from_kgrid's halving
    stays an integer divisor. Vacuum axes are NOT detected here -- callers
    force those to 1 (see _vacuum_axes in tools.dft).
    """
    b_norms = 2.0 * math.pi * np.linalg.norm(np.asarray(atoms.cell.reciprocal()), axis=1)
    grid = []
    for b in b_norms:
        # round(..., 5) guards float fuzz at exact multiples (aiida does the same)
        n = max(1, math.ceil(round(b / kdist, 5)))
        if n > 1 and n % 2 == 1:
            n += 1
        grid.append(int(n))
    return grid


def qgrid_from_kgrid(kgrid) -> list[int]:
    """Phonon q-grid derived from (and always dividing) the SCF k-grid."""
    return [k // 2 if (k % 2 == 0 and k > 4) else k for k in map(int, kgrid)]
