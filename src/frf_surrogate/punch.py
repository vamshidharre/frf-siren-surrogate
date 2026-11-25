"""Parse Nastran punch (.pch) output from SOL 111 frequency-response runs.

Supports the MAGNITUDE-PHASE (and REAL-IMAGINARY) grid-point output blocks that
Nastran writes for $DISPLACEMENT / $VELOCITY / $ACCELERATION, e.g.

    $VELOCITY
    $MAGNITUDE-PHASE OUTPUT
    $SUBCASE ID =       10048
    $POINT ID =       35101
        1.000000E+01 G      5.850444E+00      4.564463E-01      3.187709E+01
    -CONT-                  2.234876E-03      1.186267E-01      1.586428E-03
    -CONT-                  8.482604E+01      8.365439E+01      2.845480E+02
    -CONT-                  2.886703E+02      2.847940E+02      2.847940E+02

Each frequency row gives 12 values: T1 T2 T3 R1 R2 R3 for the first quantity
(magnitude or real part), then T1 T2 T3 R1 R2 R3 for the second (phase in
degrees, or imaginary part).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

COMPONENTS = ("T1", "T2", "T3", "R1", "R2", "R3")
_NUM = re.compile(r"[-+]?\d*\.\d+(?:[EeDd][-+]?\d+)?|[-+]?\d+(?:[EeDd][-+]?\d+)?")


@dataclass
class FRFBlock:
    quantity: str            # e.g. "VELOCITY"
    form: str                # "MAGNITUDE-PHASE" or "REAL-IMAGINARY"
    subcase: int
    point: int
    freq: list = field(default_factory=list)
    values: list = field(default_factory=list)   # rows of 12 floats

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.freq, float), np.asarray(self.values, float)

    def complex_response(self, component: str = "T1") -> tuple[np.ndarray, np.ndarray]:
        """Return (freq, complex H) for one component."""
        f, v = self.as_arrays()
        i = COMPONENTS.index(component.upper())
        a, b = v[:, i], v[:, 6 + i]
        if self.form.startswith("MAG"):
            h = a * np.exp(1j * np.deg2rad(b))
        else:
            h = a + 1j * b
        return f, h


def _floats(line: str) -> list[float]:
    # Punch lines end with an 8-char sequence number in columns 73-80; drop it.
    body = line[:72]
    return [float(x.replace("D", "E").replace("d", "e")) for x in _NUM.findall(body)]


def parse_punch(path: str | Path) -> list[FRFBlock]:
    """Parse every grid-point FRF block in a punch file."""
    blocks: list[FRFBlock] = []
    quantity, form, subcase = None, "MAGNITUDE-PHASE", None
    cur: FRFBlock | None = None
    row: list[float] = []

    def flush_row():
        nonlocal row
        if cur is not None and row:
            if len(row) != 13:
                raise ValueError(f"Expected 13 numbers per frequency row, got {len(row)}: {row}")
            cur.freq.append(row[0])
            cur.values.append(row[1:])
        row = []

    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("$"):
                flush_row()
                tag = line[1:72].strip()          # drop the column 73-80 sequence number
                key = tag.split("=")[0].strip().upper()
                if key in ("DISPLACEMENTS", "DISPLACEMENT", "VELOCITY", "VELOCITIES",
                           "ACCELERATION", "ACCELERATIONS"):
                    quantity = {"VELOCITIES": "VELOCITY", "DISPLACEMENTS": "DISPLACEMENT",
                                "ACCELERATIONS": "ACCELERATION"}.get(key, key)
                elif "MAGNITUDE-PHASE" in tag.upper():
                    form = "MAGNITUDE-PHASE"
                elif "REAL-IMAGINARY" in tag.upper():
                    form = "REAL-IMAGINARY"
                elif key.startswith("SUBCASE ID"):
                    subcase = int(tag.split("=")[1])
                elif key.startswith("POINT ID"):
                    cur = FRFBlock(quantity or "UNKNOWN", form, subcase or 0, int(tag.split("=")[1]))
                    blocks.append(cur)
                elif key.startswith("TITLE") or key.startswith("LABEL"):
                    # new result section: the next quantity tag will follow
                    cur = None
                continue
            if cur is None or not line.strip():
                continue
            if line.lstrip().startswith("-CONT-"):
                row.extend(_floats(line.replace("-CONT-", " ")))
            else:
                flush_row()
                # first line of a frequency row: "<freq> G <v1> <v2> <v3>"
                parts = line[:72].split()
                if len(parts) >= 2 and parts[1] in ("G", "S"):
                    row = [float(parts[0])] + _floats(" ".join(parts[2:]))
                else:
                    row = _floats(line)
        flush_row()
    return blocks


def extract_frf(path: str | Path, point: int, component: str = "T1",
                quantity: str = "VELOCITY", subcase: int | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Return (freq, complex H) for one grid point / component from a punch file."""
    for b in parse_punch(path):
        if b.point == point and b.quantity == quantity and (subcase is None or b.subcase == subcase):
            return b.complex_response(component)
    raise KeyError(f"No {quantity} block for point {point} (subcase={subcase}) in {path}")
