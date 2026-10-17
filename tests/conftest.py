"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_x():
    """n = 4, x = (1, 0, -1, 0): DFT X = (0, 2, 0, 2) (k = 0 .. 3); analytisch Z = (0, 4, 0, 0) (k = 1 doppelt, Nyquist k = 2 ist null) und z = IDFT(Z) = (1, i, -1, -i);
    Phasenschritt z[t+1] conj(z[t]) = i, Winkel pi/2, Momentanfrequenz 1/4; Hüllkurve 1."""
    return np.array([1.0, 0.0, -1.0, 0.0])
