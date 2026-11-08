"""Flujo desatendido (PC): genera datos con el profesor en el simulador, los sube a Drive y sincroniza el codigo.

  python tools/overnight.py --round 0                 # profesor limpio+ruido en las 21 pistas
  python tools/overnight.py --round 1 --student agents/bc_cnn/weights/model.pt   # DAgger con el estudiante

Es reanudable (salta carreras ya subidas) y deja un informe en results/overnight_<ronda>.md.
Los episodios se suben a Drive (apex_kart_processed/teacher_r<ronda>) y se borran del disco local.
Para dejarlo corriendo aunque se cierre la sesion: lanzarlo con Start-Process (ver docs/COLAB.md).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "gmail:UTEC/DEEP/apex_kart_processed"


def done_remote(dest):
    r = subprocess.run(["rclone", "lsf", "-R", "--include", "info.json", dest], capture_output=True, text=True, encoding="utf-8")
    return {l.rsplit("/info.json", 1)[0] for l in r.stdout.splitlines() if l.endswith("info.json")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--student")
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--laps", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--skip-tracks", nargs="*", default=[])
    a = p.parse_args()
    tracks = [t for t in json.loads((ROOT / "results" / "assets.json").read_text())["tracks"] if t not in a.skip_tracks]
    name = f"teacher_r{a.round}"
    dest = f"{REMOTE}/{name}"
    already = done_remote(dest)
    tag = f"t{a.round}"
    todo_tracks = [t for t in tracks if any(f"{tag}_{t}_s{s}" not in already for s in a.seeds)]
    seeds = a.seeds
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {len(tracks)} pistas x {len(seeds)} semillas; ya subidas: {len(already)}", flush=True)
    cmd = [sys.executable, "-u", str(ROOT / "training" / "dagger.py"), "--out", str(ROOT / "data" / name), "--tracks", *todo_tracks,
           "--seeds", *map(str, seeds), "--laps", str(a.laps), "--noise", str(a.noise), "--kart", "cycle", "--tag", tag,
           "--workers", str(a.workers), "--upload", dest, "--max-seconds", "330"]
    if a.student:
        cmd += ["--student", a.student, "--beta", str(a.beta)]
    subprocess.run(cmd, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "tools" / "apex.py"), "sync"], cwd=ROOT)
    final = done_remote(dest)
    subprocess.run(["rclone", "rcat", f"{dest}/_DONE"], input=str(len(final)), text=True)  # avisa a Colab de que ya hay datos completos
    lines = [f"# Informe {name}", "", f"- Episodios subidos a `{dest}`: {len(final)}", ""]
    (ROOT / "results" / f"overnight_{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] terminado: {len(final)} episodios en Drive", flush=True)


if __name__ == "__main__":
    main()
