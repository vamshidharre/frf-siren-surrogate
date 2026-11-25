"""Build a HyperStudy Templex template (.tpl) from Nastran MAT1 cards.

Every MAT1 card gets three design variables: E_<mid>, RHO_<mid>, GE_<mid>.
Fields are read and written on the 8-character small-field boundaries, so the
column layout of the deck is preserved.

MAT1 small-field layout (8 chars each):
    1 MAT1 | 2 MID | 3 E | 4 G | 5 NU | 6 RHO | 7 A | 8 TREF | 9 GE | 10 cont
"""

from __future__ import annotations

from pathlib import Path

E_FMT, RHO_FMT, GE_FMT = "%8.0f", "%8.2E", "%8.3f"


def _fields(line: str) -> list[str]:
    line = line.ljust(80)
    return [line[i:i + 8] for i in range(0, 80, 8)]


def mat1_to_templex(lines: list[str]) -> tuple[list[str], list[str]]:
    out, variables = [], []
    for line in lines:
        if not line.startswith("MAT1") or line.startswith("MAT1*"):
            out.append(line.rstrip("\n"))
            continue
        f = _fields(line.rstrip("\n"))
        mid = f[1].strip()
        if not mid.isdigit():
            raise ValueError(f"Unexpected MID field {f[1]!r} in line: {line!r}")
        f[2] = '{E_%s,"%s"}' % (mid, E_FMT)
        f[5] = '{RHO_%s,"%s"}' % (mid, RHO_FMT)
        f[8] = '{GE_%s,"%s"}' % (mid, GE_FMT)
        variables += [f"E_{mid}", f"RHO_{mid}", f"GE_{mid}"]
        out.append("".join(f).rstrip())
    return out, variables


def write_template(src: str | Path, dst: str | Path) -> list[str]:
    lines = Path(src).read_text().splitlines()
    out, variables = mat1_to_templex(lines)
    Path(dst).write_text("\n".join(out) + "\n")
    return variables
