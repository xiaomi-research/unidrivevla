"""Bootstrap sys.path for bundled 3rd_party directories that lack pip installs.

Runs once on the first import of `lead.*`. Prepends paths to sys.path without
touching os.environ, so any caller-set PYTHONPATH is preserved. This avoids
needing PYTHONPATH exports for notebooks / IDEs that spawn a Python process
directly (Jupyter, PyCharm, VSCode, plain `python`, pytest, etc.).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_THIRD_PARTY_PATHS = []
if os.environ.get("CARLA_ROOT"):
    _carla_pythonapi = Path(os.environ["CARLA_ROOT"]) / "PythonAPI/carla"
    _carla_eggs = sorted(
        (_carla_pythonapi / "dist").glob("carla-0.9.15-py3.*-linux-x86_64.egg"),
        reverse=True,
    )
    _THIRD_PARTY_PATHS.extend(_carla_eggs)
    _THIRD_PARTY_PATHS.append(_carla_pythonapi)
_THIRD_PARTY_PATHS.append(_REPO_ROOT / "3rd_party/CARLA_0915/PythonAPI/carla")

_found_carla_pythonapi = False
for _p in _THIRD_PARTY_PATHS:
    _s = str(_p)
    if not _p.exists():
        continue
    _found_carla_pythonapi = True
    if _s not in sys.path:
        sys.path.insert(-1, _s)

if not _found_carla_pythonapi:
    print(
        "3rd-party CARLA PythonAPI path not found. Set CARLA_ROOT to a CARLA "
        f"root containing PythonAPI/carla, or install {_THIRD_PARTY_PATHS[-1]}.",
    )
