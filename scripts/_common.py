import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)
