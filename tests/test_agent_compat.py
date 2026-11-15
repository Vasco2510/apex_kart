"""Prueba sin simulador: el agente carga checkpoints viejos (huber) y nuevos (bins + aux) y devuelve acciones validas.
  python tests/test_agent_compat.py [ruta_checkpoint ...]   (por defecto los pesos locales que existan)"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agents" / "bc_cnn"))
from agent import Agent  # noqa: E402
from model import Net, steer_from_bins  # noqa: E402


def check(path):
    ag = Agent({"checkpoint": str(path), "device": "cpu", "unstick": False})
    ag.reset()
    rng = np.random.default_rng(0)
    for _ in range(12):
        a = ag.act(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8))
        assert -1 <= a.steer <= 1
    print("OK", path, "| steer_bins=%d n_aux=%d" % (ag.net.steer_bins, ag.net.n_aux), "| ultimo steer %.3f" % a.steer)


def unit():
    n = Net(steer_bins=21, n_aux=3)
    x = torch.rand(2, 9, 90, 160)
    assert n(x).shape == (2, 6)          # salida de conduccion
    n.full = True
    assert n(x).shape == (2, 21 + 5 + 3)  # salida completa
    lg = torch.full((1, 21), -9.0)
    lg[0, 17] = 9.0                       # bin 17 -> 0.7
    assert abs(float(torch.tanh(steer_from_bins(lg))[0, 0]) - 0.7) < 0.02
    lg[0, 3] = 9.0                        # dos modas opuestas: NO promedia a 0
    assert abs(float(torch.tanh(steer_from_bins(lg))[0, 0])) > 0.5
    print("OK unit bins")


if __name__ == "__main__":
    unit()
    paths = sys.argv[1:] or sorted(ROOT.glob("agents/bc_cnn/weights/*/model.pt"))
    for p in paths:
        check(p)
