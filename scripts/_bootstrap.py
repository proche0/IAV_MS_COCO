"""Rend ``coco_mlc`` importable sans installation prealable.

Les scripts commencent tous par ``import _bootstrap``. Le paquet reste
installable proprement (``pip install -e .``) pour les notebooks et Colab.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
