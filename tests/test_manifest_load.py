"""El manifiesto de entrega carga en el worker real (subproceso) y responde a act() sin simulador.
  python tests/test_manifest_load.py [agents/bc_cnn/final/agent.json]"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from stk_agent.agents import AgentWorker

m = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "agents/bc_cnn/final/agent.json")
with tempfile.TemporaryDirectory() as t:
    w = AgentWorker(m, Path(t) / "agent_0.log")
    print("cargado:", w.name)
    w.close()
print("OK", m)
