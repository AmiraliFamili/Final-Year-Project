"""Interpretability & explainability implementations.

Modules here do `from _shared import ...`. The bootstrap below puts the
project root on sys.path so that import resolves regardless of how this
package is loaded.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
