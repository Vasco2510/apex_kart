"""Genera datos con el profesor privilegiado en el simulador (formato igual a tools/preprocess.py).

Ronda 0 (profesor conduce, con ruido para visitar estados de recuperacion; etiqueta = accion limpia del profesor):
  python training/dagger.py --out data/teacher_r0 --tracks lighthouse candela_city --seeds 1 2 --laps 2 --noise 0.3
Ronda k>=1 (DAgger: el estudiante conduce parte del tiempo y el profesor etiqueta lo que el estudiante vivio):
  python training/dagger.py --out data/teacher_r1 --student agents/bc_cnn/weights/model.pt --beta 0.3 ...

Cada carrera es un episodio <out>/<id>/ con frames.npy (160x90, stride 2), labels.npz e info.json (ultimo = completo).
Reanudable: salta los episodios que ya tienen info.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "agents" / "bc_cnn"))

AUX_KEYS = ("speed", "distance_down_track", "overall_distance", "position", "finished_laps",
            "energy", "is_on_road", "jumping", "has_finished_race")
RACE = ("GO_PHASE", "MUSIC_PHASE", "RACE_PHASE")
KARTS = ["adiumy", "amanda", "beastie", "emule", "gavroche", "gnu", "hexley", "kiki", "konqi", "nolok", "pidgin",
         "puffy", "sara_the_racer", "sara_the_wizard", "suzanne", "tux", "wilber", "xue"]


def shrink(rgb, down):
    h, w, c = rgb.shape
    return rgb.reshape(h // down, down, w // down, down, c).mean(axis=(1, 3)).astype(np.uint8)


def _rollout(job):
    track, seed, a = job
    name = f"{a['tag']}_{track}_s{seed}"
    out = Path(a["out"]) / name
    if (out / "info.json").exists():
        return name, "skip"
    from stk_agent.config import RaceSettings
    from stk_agent.gym_env import STKPixelEnv
    from teacher import Teacher
    rng = np.random.default_rng(seed * 1009 + zlib.crc32(track.encode()) % 1000)
    student = None
    if a["student"]:
        from agent import Agent
        import torch
        torch.set_num_threads(1)  # varios procesos en paralelo: evitar pelea por los nucleos
        student = Agent({"checkpoint": a["student"], "device": "cpu"})
    kart = a["kart"] if a["kart"] != "cycle" else KARTS[(zlib.crc32(name.encode())) % len(KARTS)]
    cfg = RaceSettings(track=track, kart=kart, laps=a["laps"], num_karts=a["num_karts"], seed=seed,
                       player_slot=0, max_seconds=a["max_seconds"], difficulty=a["difficulty"])
    env = STKPixelEnv(cfg, privileged=True)
    obs, info = env.reset(seed=seed)
    teacher = Teacher(env.base.track)
    frames, acts, aux, phases = [], [], {k: [] for k in AUX_KEYS}, []
    ou, use_teacher, step = 0.0, True, 0
    done, last = False, None
    t0 = time.time()
    while not done:
        st = info["privileged"]
        k = st["karts"][0]
        phase = st["phase"]
        lab = teacher.action(k, in_race=phase in RACE)
        if student is not None:
            stu = student.act(obs)  # mantiene el buffer de frames al dia aunque no se use
            if step % 60 == 0:  # se decide quien conduce por tramos de ~2 s
                use_teacher = rng.random() < a["beta"]
        if student is None or use_teacher:
            ex = dict(lab)
            if a["noise"] > 0 and phase in RACE:  # ruido de giro correlacionado (Ornstein-Uhlenbeck)
                ou += -0.05 * ou + a["noise"] * 0.25 * rng.standard_normal()
                ex["steer"] = float(np.clip(ex["steer"] + ou, -1, 1))
        else:
            ex = stu.as_dict()
        if step % a["stride"] == 0:
            frames.append(shrink(obs, a["down"]))
            acts.append([lab["steer"], lab["acceleration"], float(lab["brake"]), float(lab["nitro"]),
                         float(lab["drift"]), float(lab["fire"]), float(lab["rescue"])])
            for key in AUX_KEYS:
                aux[key].append(float(k.get(key, 0.0)))
            phases.append(phase)
        act = {"steer": np.array([ex["steer"]], np.float32), "acceleration": np.array([float(ex["acceleration"])], np.float32),
               **{n: int(ex[n]) for n in ("brake", "nitro", "drift", "fire", "rescue")}}
        obs, r, term, trunc, info = env.step(act)
        step += 1
        done = term or trunc
        last = info["privileged"]["karts"][0]
    env.close()
    names = sorted(set(phases))
    n = len(frames)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "frames.npy", np.stack(frames))
    A = np.asarray(acts, np.float32)
    np.savez(out / "labels.npz", actions=A, actions_mean=A, phase_id=np.array([names.index(p) for p in phases], np.int8),
             rgb_index=np.arange(n, dtype=np.int64) * a["stride"],
             **{f"aux_{k}": np.asarray(v, np.float32) for k, v in aux.items()})
    result = {"finished": bool(last["has_finished_race"]), "finish_time": float(last["finish_time"]),
              "overall_distance": float(last["overall_distance"]), "native_position": int(last["position"])}
    info_json = {"episode_id": name, "track": track, "kart": kart, "difficulty": a["difficulty"], "laps": a["laps"],
                 "num_karts": a["num_karts"], "reverse": False, "seed": seed, "human_slot": 0, "group_id": "teacher",
                 "driver_id": a["tag"], "source": "teacher", "student": bool(a["student"]), "beta": a["beta"],
                 "noise": a["noise"], "seconds": step / 30.0, "num_frames": n, "phases": names, "down": a["down"],
                 "stride": a["stride"], "frame_shape": list(frames[0].shape), "result": result}
    (out / "info.json").write_text(json.dumps(info_json, indent=1), encoding="utf-8")
    msg = f"ok {n} frames, fin={result['finished']} t={result['finish_time']:.0f}s dist={result['overall_distance']:.0f} wall={time.time() - t0:.0f}s"
    if a.get("upload"):  # sube el episodio a Drive y libera el disco local
        r = subprocess.run(["rclone", "move", str(out), f"{a['upload']}/{name}"], capture_output=True, text=True)
        msg += " | subido" if r.returncode == 0 else f" | ERROR subiendo: {r.stderr[-200:]}"
        shutil.rmtree(out, ignore_errors=True) if r.returncode == 0 else None
    return name, msg


def rollout(job):
    """Una carrera aislada: si el motor nativo cae, reintenta una vez y si no, informa sin tumbar el lote."""
    for intento in (1, 2):
        try:
            return _rollout(job)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    return f"{job[0]}_s{job[1]}", f"FALLO tras 2 intentos ({err})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--laps", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.3, help="0 = profesor limpio")
    p.add_argument("--student", help="checkpoint del estudiante (activa DAgger)")
    p.add_argument("--beta", type=float, default=0.3, help="fraccion de tiempo que conduce el profesor en DAgger")
    p.add_argument("--tag", default="t")
    p.add_argument("--kart", default="tux", help="un kart o `cycle` para variar entre los 18 segun la carrera")
    p.add_argument("--upload", help="destino rclone, p. ej. gmail:UTEC/DEEP/apex_kart_processed/teacher_r0")
    p.add_argument("--num-karts", type=int, default=8)
    p.add_argument("--difficulty", type=int, default=2)
    p.add_argument("--max-seconds", type=float, default=300.0)
    p.add_argument("--down", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    cfg = {k: v for k, v in vars(a).items() if k not in ("tracks", "seeds", "workers")}
    cfg["student"] = str(Path(a.student).resolve()) if a.student else None
    jobs = [(t, s, cfg) for t in a.tracks for s in a.seeds]
    print(f"{len(jobs)} carreras", flush=True)
    if os.name == "nt":  # que Windows no suspenda el equipo mientras corre
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    t0 = time.time()
    res = map(rollout, jobs) if a.workers <= 1 else ProcessPoolExecutor(a.workers).map(rollout, jobs)
    for i, (name, status) in enumerate(res, 1):
        print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f}s {name} {status}", flush=True)


if __name__ == "__main__":
    main()
