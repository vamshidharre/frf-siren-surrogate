"""Create a HyperStudy Templex template from a file of Nastran MAT1 cards.

    python scripts/make_tpl.py mat1_cards.txt MAT1_template.tpl
"""

import argparse

import _common  # noqa: F401
from frf_surrogate.templex import write_template

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("src", help="text file containing the MAT1 cards (small-field format)")
p.add_argument("dst", help="output .tpl path")
a = p.parse_args()

v = write_template(a.src, a.dst)
print(f"wrote {a.dst}: {len(v)} design variables ({len(v) // 3} materials x E, RHO, GE)")
