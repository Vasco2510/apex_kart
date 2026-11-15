"""Construye la carpeta de entrega autocontenida agents/bc_cnn/final/ (agent.json + agent.py + model.py + model.pt + README).

  python tools/make_final.py bc_teacher_r2_bins                      # usa weights/<run>/model.pt (mejor por val)
  python tools/make_final.py bc_teacher_r2_bins --weights model_last.pt --name apex_kart_v2

La carpeta se versiona en git (unico lugar con pesos permitidos por .gitignore). El profesor ejecuta:
  python -m stk_agent play agents/bc_cnn/final/agent.json --config configs/evaluation.json --out runs/x
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AG = ROOT / "agents" / "bc_cnn"
FINAL = AG / "final"

README = """# Agente apex_kart ({name})

Agente visual para SuperTuxKart: entra RGB 640x360 con HUD, sale `Action`. Sin acceso a estado privilegiado.
Red CNN (3 frames apilados, 160x90) entrenada por clonacion de comportamiento + DAgger (run `{run}`, pesos `{weights}`).
Incluye regla anti-atasco (si la imagen no cambia ~2 s, rescata).

## Uso
    pip install -e <repo>            # libreria stk_agent (+ torch, numpy)
    python -m stk_agent play agents/bc_cnn/final/agent.json --config configs/evaluation.json --out runs/prueba
    python -m stk_agent race agents/bc_cnn/final/agent.json otro/agent.json --out runs/carrera

`device` en agent.json: `cpu` por defecto (portable; ~12-20 ms/paso); `auto` usa GPU si existe.
Ajustes opcionales en `config`: `threshold` (0.5), `unstick` (true), `still_thr`, `rescue_after`.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--weights", default="model.pt")
    p.add_argument("--name", default="apex_kart_bc")
    a = p.parse_args()
    src = AG / "weights" / a.run / a.weights
    if not src.exists():
        raise SystemExit(f"falta {src}; ejecuta: python tools/apex.py pull {a.run}")
    FINAL.mkdir(exist_ok=True)
    for f in ("agent.py", "model.py"):
        shutil.copyfile(AG / f, FINAL / f)
    shutil.copyfile(src, FINAL / "model.pt")
    manifest = {"name": a.name, "entry": "agent.py", "class": "Agent",
                "config": {"checkpoint": "model.pt", "device": "cpu", "threshold": 0.5}}
    (FINAL / "agent.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (FINAL / "README.md").write_text(README.format(name=a.name, run=a.run, weights=a.weights), encoding="utf-8")
    print("entrega lista en", FINAL, f"({(FINAL / 'model.pt').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
