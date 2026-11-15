"""geo_labels sobre una pista sintetica (recta a lo largo de +Z con curva): signos y magnitudes razonables."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from dagger import geo_labels  # noqa: E402
from teacher import Teacher  # noqa: E402

n = 200
z = np.arange(n, dtype=float) * 2.0  # nodos cada 2 m, recta hacia +Z
nodes = np.zeros((n, 2, 3))
nodes[:, 0, 0], nodes[:, 1, 0] = -4, 4
nodes[:, :, 2] = z[:, None]
track = {"path_nodes": nodes, "successor_offsets": np.arange(n + 1), "successor_values": (np.arange(n) + 1) % n,
         "path_width": np.full(n, 8.0)}
t = Teacher(track)
k = {"location": [2.0, 0.0, 20.0], "front": [2.0, 0.0, 21.0], "node": 10}  # 2 m a la derecha (+X), apuntando recto
g = geo_labels(t, k)
assert abs(g["lat"] - 0.5) < 1e-6 and abs(g["hdg"]) < 1e-6, g
assert g["ang5"] < 0 and g["ang40"] < 0, g          # el objetivo esta a la izquierda (-X) del kart
assert abs(g["ang5"]) > abs(g["ang40"]) and g["curve"] > 0, g
print("OK geo_labels", {a: round(b, 3) for a, b in g.items()})
