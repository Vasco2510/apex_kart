"""Progreso de la generacion de datos de una ronda (lee logs/overnight_r<N>.log).

  python tools/progreso.py          # ronda 1
  python tools/progreso.py 2        # ronda 2
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    n = sys.argv[1] if len(sys.argv) > 1 else "1"
    log = ROOT / "logs" / f"overnight_r{n}.log"
    if not log.exists():
        sys.exit(f"no existe {log}")
    lines = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if re.match(r"^\[\d+/\d+\]", l)]
    if not lines:
        sys.exit("todavia no hay carreras terminadas")
    done, total = map(int, re.match(r"^\[(\d+)/(\d+)\]", lines[-1]).groups())
    fails = [l for l in lines if "FALLO" in l]
    secs = float(re.match(r"^\[\d+/\d+\] (\d+)s", lines[-1]).group(1))
    eta = secs / done * (total - done) / 60 if done else 0
    age = (time.time() - log.stat().st_mtime) / 60
    print(f"Ronda {n}: {done}/{total} carreras ({done / total:.0%}) | fallidas: {len(fails)} | ETA ~{eta:.0f} min | ultima salida hace {age:.0f} min")
    print("ultima:", lines[-1][:110])
    for f in fails:
        print("  FALLO:", f[:100])
    r = subprocess.run(["rclone", "lsf", "--dirs-only", f"gmail:UTEC/DEEP/apex_kart_processed/teacher_r{n}"], capture_output=True, text=True)
    print(f"Episodios ya en Drive: {len([x for x in r.stdout.splitlines() if x.strip()])}")


if __name__ == "__main__":
    main()
