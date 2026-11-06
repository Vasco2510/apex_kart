"""Atajos del flujo PC <-> Drive <-> Colab (usa rclone con el remote `gmail`).

  python tools/apex.py sync            # sube el codigo a Drive (Colab lo copia desde ahi)
  python tools/apex.py status          # lista corridas en Drive y su ultima epoca
  python tools/apex.py pull bc_v1      # baja model.pt y log.json de la corrida a agents/bc_cnn/weights y runs/
  python tools/apex.py eval bc_v1      # evalua el agente con el arnés (carpeta runs/eval_bc_v1_<fecha>)

Rutas en Drive (cuenta Gmail, remote rclone `gmail`):
  UTEC/DEEP/MarioKart_copia      datos crudos del profesor (copia)
  UTEC/DEEP/apex_kart_code       codigo (lo escribe `sync`)
  UTEC/DEEP/apex_kart_processed  datos preprocesados (los escribe el notebook 01)
  UTEC/DEEP/apex_kart_runs       corridas de entrenamiento (las escribe el notebook 02)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "gmail:UTEC/DEEP"
EXCLUDE = [".git/**", "data/**", "runs/**", "contexto/**", "dist/**", "*.zip", "__pycache__/**", "*.egg-info/**",
           "weights/**", "*.log", "results/preview/**", ".venv/**", "*.pt"]


def rclone(*args, capture=False):
    cmd = ["rclone", *args]
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8").stdout
    subprocess.run(cmd, check=True)


def sync(a):
    flags = [f"--exclude={e}" for e in EXCLUDE] + ["-P"] + (["--dry-run"] if a.dry_run else [])
    rclone("sync", str(ROOT), f"{REMOTE}/apex_kart_code", *flags)


def status(a):
    out = rclone("lsf", f"{REMOTE}/apex_kart_runs", "--dirs-only", capture=True)
    runs = [l.strip("/ \n") for l in out.splitlines() if l.strip()]
    if not runs:
        print("Sin corridas todavia en", f"{REMOTE}/apex_kart_runs")
    for r in runs:
        txt = rclone("cat", f"{REMOTE}/apex_kart_runs/{r}/log.json", capture=True)
        try:
            log = json.loads(txt)
            last = log[-1]
            print(f"{r}: epoca {last['epoch']}, train {last['train_loss']:.4f}",
                  f"val {last['val']['loss']:.4f}" if "val" in last else "")
        except Exception:
            print(f"{r}: sin log.json (en curso o sin empezar)")


def pull(a):
    dst = ROOT / "agents" / a.agent / "weights"
    dst.mkdir(parents=True, exist_ok=True)
    rclone("copyto", f"{REMOTE}/apex_kart_runs/{a.run}/model.pt", str(dst / "model.pt"), "-P")
    (ROOT / "runs" / a.run).mkdir(parents=True, exist_ok=True)
    rclone("copyto", f"{REMOTE}/apex_kart_runs/{a.run}/log.json", str(ROOT / "runs" / a.run / "log.json"))
    print("modelo en", dst / "model.pt")


def evaluate(a):
    out = ROOT / "runs" / f"eval_{a.run}_{time.strftime('%m%d_%H%M')}"
    subprocess.run([sys.executable, str(ROOT / "tools" / "evaluate.py"), str(ROOT / "agents" / a.agent / "agent.json"),
                    "--out", str(out)], check=True)


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("sync"); x.add_argument("--dry-run", action="store_true"); x.set_defaults(f=sync)
    s.add_parser("status").set_defaults(f=status)
    for name, f in (("pull", pull), ("eval", evaluate)):
        x = s.add_parser(name); x.add_argument("run"); x.add_argument("--agent", default="bc_cnn"); x.set_defaults(f=f)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
