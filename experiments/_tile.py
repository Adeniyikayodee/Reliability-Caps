"""Load the locally cached TESSERA tile and dequantise it."""
from __future__ import annotations
import pathlib, sys
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
TILE = REPO / "global_0.1_degree_representation/2024/grid_3.35_6.45"


def load_tile() -> np.ndarray:
    q = np.load(TILE / "grid_3.35_6.45.npy")
    s = np.load(TILE / "grid_3.35_6.45_scales.npy")
    return (q.astype("float32") * s[:, :, None]).astype("float32")
