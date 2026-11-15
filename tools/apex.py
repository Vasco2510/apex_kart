"""Atajos del flujo PC <-> Drive <-> Colab (usa rclone con el remote `gmail`).

  python tools/apex.py sync            # sube el codigo a Drive (Colab lo copia desde ahi)
  python tools/apex.py status          # lista corridas en Drive y su ultima epoca
  python tools/apex.py pull bc_v1                 # baja model.pt (mejor por error de giro) y log.json
  python tools/apex.py pull bc_v1 --file ep20.pt  # baja una instantanea concreta (ep05.pt, ep10.pt, last.pt ...)
  python tools/apex.py eval bc_v1 --weights ep20.pt --tracks lighthouse candela_city   # evalua ese checkpoint

Rutas en Drive (cuenta Gmail, remote rclone `gmail`):
  UTEC/DEEP/MarioKart_copia      datos crudos del profesor (copia)
  UTEC/DEEP/apex_kart_code       codigo (lo escribe `sync`)
  UTEC/DEEP/apex_kart_processed  datos preprocesados (los escribe el notebook 01)
  UTEC/DEEP/apex_kart_runs       corridas de entrenamiento (las escribe el notebook 02)
"""
import argparse
import json
import shutil
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
    """Pesos de cada corrida en su carpeta: agents/<agent>/weights/<run>/<file>. model.pt sin sufijo = el ultimo traido."""
    dst = ROOT / "agents" / a.agent / "weights" / a.run
    dst.mkdir(parents=True, exist_ok=True)
    rclone("copyto", f"{REMOTE}/apex_kart_runs/{a.run}/{a.file}", str(dst / a.file), "-P")
    (ROOT / "runs" / a.run).mkdir(parents=True, exist_ok=True)
    rclone("copyto", f"{REMOTE}/apex_kart_runs/{a.run}/log.json", str(ROOT / "runs" / a.run / "log.json"))
    if a.file == "model.pt":  # copia "actual" para quien use el manifiesto por defecto
        shutil.copyfile(dst / a.file, dst.parent / "model.pt")
    print("modelo en", dst / a.file)


def evaluate(a):
    """Evalua SIEMPRE los pesos de la corrida indicada (manifiesto temporal) y deja un resumen versionable en results/evals/."""
    stem = Path(a.weights).stem
    out = ROOT / "runs" / f"eval_{a.run}_{stem}_{time.strftime('%m%d_%H%M')}"
    base = ROOT / "agents" / a.agent / "agent.json"
    ckpt = f"weights/{a.run}/{a.weights}"
    if not (base.parent / ckpt).exists():
        sys.exit(f"no existe {base.parent / ckpt}: ejecuta  python tools/apex.py pull {a.run}")
    cfg = json.loads(base.read_text(encoding="utf-8"))
    cfg["name"] = f"{cfg['name']}_{a.run}_{stem}"
    cfg["config"]["checkpoint"] = ckpt
    manifest = base.with_name(f"agent_{a.run}_{stem}.json")
    manifest.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "tools" / "evaluate.py"), str(manifest), "--out", str(out)]
    if a.tracks:
        cmd += ["--tracks", *a.tracks]
    if a.seeds:
        cmd += ["--seeds", *map(str, a.seeds)]
    subprocess.run(cmd, check=True)
    keep = ROOT / "results" / "evals"
    keep.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        shutil.copyfile(out / "summary.json", keep / f"{out.name}.json")
        print("resumen guardado en", keep / f"{out.name}.json")


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("sync"); x.add_argument("--dry-run", action="store_true"); x.set_defaults(f=sync)
    s.add_parser("status").set_defaults(f=status)
    x = s.add_parser("pull"); x.add_argument("run"); x.add_argument("--agent", default="bc_cnn")
    x.add_argument("--file", default="model.pt"); x.set_defaults(f=pull)
    x = s.add_parser("eval"); x.add_argument("run"); x.add_argument("--agent", default="bc_cnn")
    x.add_argument("--weights", default="model.pt"); x.add_argument("--tracks", nargs="*"); x.add_argument("--seeds", nargs="*", type=int)
    x.set_defaults(f=evaluate)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
