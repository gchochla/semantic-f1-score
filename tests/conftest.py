from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.append(str(root))
