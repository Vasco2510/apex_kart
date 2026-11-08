"""Pipeline de punta a punta (pensado para Colab; el notebook solo llama a este archivo).

Etapas (se saltan solas si ya estan hechas, asi que re-ejecutar reanuda):
  preprocess  episodios crudos (Drive) -> formato compacto en Drive      [CPU, una vez por subconjunto]
  fetch       copia lo preprocesado de Drive al disco local de la VM     [rapido para entrenar]
  smoke       entrenamiento de 1 epoca minima; aborta si algo falla
  train       entrenamiento real; guarda en Drive cada epoca y reanuda solo

  python training/pipeline.py --preset v1 --persist /content/drive/MyDrive/UTEC/DEEP
  python training/pipeline.py --preset v1 --persist ... --stages preprocess      # en runtime CPU (no gasta GPU)
  python training/pipeline.py --preset v1 --persist ... --epochs 20              # extras se pasan a train_bc.py

El preset es configs/train/<preset>.json (claves: subset, processed, run + las de train_bc.py).
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("preprocess", "fetch", "smoke", "train")


def sh(cmd):
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def count_done(d):
    return len(list(Path(d).glob("*/info.json"))) if Path(d).exists() else 0


def stage_preprocess(a, cfg, raw, processed, expected):
    have = count_done(processed)
    if have >= expected:
        print(f"[preprocess] ya hecho: {have}/{expected} episodios", flush=True)
        return
    print(f"[preprocess] {have}/{expected} listos; procesando el resto", flush=True)
    sh([sys.executable, "-u", "tools/preprocess.py", "--root", raw, "--list", a.subset_path, "--out", processed,
        "--down", cfg.get("down", 4), "--stride", cfg.get("stride", 2), "--workers", os.cpu_count() or 2])
    have = count_done(processed)
    if have < expected * 0.98:
        sys.exit(f"[preprocess] solo {have}/{expected} episodios; vuelve a ejecutar para reintentar")
    if have < expected:
        print(f"[preprocess] AVISO: {expected - have} episodios omitidos por error; se sigue con {have}", flush=True)


def stage_fetch(processed, local):
    local.mkdir(parents=True, exist_ok=True)
    todo = [d.parent for d in sorted(Path(processed).glob("*/info.json")) if not (local / d.parent.name / "info.json").exists()]
    print(f"[fetch] {len(todo)} episodios por copiar a {local}", flush=True)
    t0 = time.time()
    for i, src in enumerate(todo, 1):
        tmp = local / (src.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src, tmp)
        os.replace(tmp, local / src.name)  # el directorio aparece completo o no aparece
        if i % 20 == 0 or i == len(todo):
            print(f"[fetch] {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


def stage_smoke(a, cfg, local, scratch):
    out = scratch / "smoke"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [sys.executable, "-u", "training/train_bc.py", "--config", a.config_path, "--data", local, "--out", out,
           "--epochs", 1, "--samples-per-epoch", 1024, "--batch", 64, "--workers", 0, "--val-tracks"]
    sh(cmd)
    log = json.loads((out / "log.json").read_text())
    loss = log[-1]["train_loss"]
    if not math.isfinite(loss):
        sys.exit("[smoke] la perdida no es finita; no se lanza el entrenamiento largo")
    print(f"[smoke] OK (perdida {loss:.3f}, {log[-1]['samples_per_s']} muestras/s)", flush=True)


def stage_train(a, cfg, local, run_dir, extra):
    sh([sys.executable, "-u", "training/train_bc.py", "--config", a.config_path, "--data", local, "--out", run_dir,
        "--max-minutes", a.max_minutes, *extra])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="v1")
    p.add_argument("--persist", required=True, help="carpeta base en Drive, p. ej. /content/drive/MyDrive/UTEC/DEEP")
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--local", help="disco local para los datos (por defecto /content/processed_<nombre>)")
    p.add_argument("--subset", help="lista de episodios (por defecto la del preset)")
    p.add_argument("--max-minutes", type=float, default=150)
    p.add_argument("--allow-cpu", action="store_true")
    a, extra = p.parse_known_args()
    a.config_path = ROOT / "configs" / "train" / f"{a.preset}.json"
    cfg = json.loads(a.config_path.read_text(encoding="utf-8"))
    a.subset_path = Path(a.subset) if a.subset else (ROOT / cfg["subset"] if cfg.get("subset") else None)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f"etapas desconocidas: {bad}; validas: {STAGES}")
    persist = Path(a.persist)
    raw = persist / "MarioKart_copia"
    processed = persist / "apex_kart_processed" / cfg["processed"]
    run_dir = persist / "apex_kart_runs" / cfg["run"]
    local = Path(a.local or f"/content/processed_{cfg['processed']}")
    expected = sum(1 for l in a.subset_path.read_text(encoding="utf-8").splitlines() if l.strip()) if a.subset_path else count_done(processed)
    if a.subset_path is None and expected == 0:
        sys.exit(f"no hay episodios en {processed}; genera los datos del profesor primero (tools/overnight.py)")
    print(f"preset={a.preset} etapas={stages} episodios={expected}\n  crudos={raw}\n  procesados={processed}\n  corrida={run_dir}", flush=True)

    if "preprocess" in stages and a.subset_path:
        stage_preprocess(a, cfg, raw, processed, expected)
    if {"smoke", "train"} & set(stages):
        import torch
        if not torch.cuda.is_available() and not a.allow_cpu:
            sys.exit("SIN GPU: cambia el runtime a GPU (T4) o usa --allow-cpu para una prueba")
    if "fetch" in stages:
        stage_fetch(processed, local)
    if "smoke" in stages:
        stage_smoke(a, cfg, local, local.parent / "_scratch")
    if "train" in stages:
        stage_train(a, cfg, local, run_dir, extra)
        print(f"\nListo. En tu PC:  python tools/apex.py pull {cfg['run']}", flush=True)


if __name__ == "__main__":
    main()
