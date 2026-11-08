"""Profesor con informacion privilegiada (solo para generar etiquetas; el agente final NO la tiene).

Sigue la linea de conducción de la pista (path_nodes) con persecucion pura (pure pursuit) y controla velocidad,
nitro y rescate. Funciona en cualquier pista porque el simulador entrega su geometria.

  python training/teacher.py --tracks lighthouse candela_city --laps 1
"""
import argparse
import math
import sys
import time

import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Teacher:
    def __init__(self, track, steer_sign=1.0, lookahead=(7.0, 0.55, 30.0), steer_gain=1.0, drift=False, drift_curve=0.4,
                 drift_speed=12.0, brake_curve=0.9, brake_speed=14.0, nitro_err=0.08, nitro_curve=0.25):
        self.centers = np.asarray(track["path_nodes"], np.float64).mean(axis=1)  # (N, 3)
        self.offs = np.asarray(track["successor_offsets"])
        self.vals = np.asarray(track["successor_values"])
        self.width = np.asarray(track["path_width"], np.float64)
        self.sign = steer_sign
        self.la0, self.la_k, self.la_max = lookahead
        self.gain = steer_gain
        self.drift, self.drift_curve, self.drift_speed = drift, drift_curve, drift_speed
        self.brake_curve, self.brake_speed = brake_curve, brake_speed
        self.nitro_err, self.nitro_curve = nitro_err, nitro_curve
        self.stuck = 0
        self.cool = 0  # pasos de espera tras un rescate
        self.pulse = 0  # pasos restantes con el boton de rescate pulsado
        # primer sucesor de cada nodo (si hay bifurcaciones se sigue la rama 0)
        n = len(self.centers)
        self.nxt = np.array([self._succ(i) for i in range(n)])

    def _succ(self, i):
        a, b = self.offs[i], self.offs[i + 1] if i + 1 < len(self.offs) else self.offs[i]
        return int(self.vals[a]) if b > a else (i + 1) % len(self.centers)

    def reset(self):
        self.stuck = self.cool = self.pulse = 0

    def target(self, node, dist):
        """Centro del nodo alcanzado tras avanzar `dist` metros por la linea."""
        i, acc = node, 0.0
        for _ in range(len(self.centers)):
            j = int(self.nxt[i])
            acc += float(np.linalg.norm(self.centers[j] - self.centers[i]))
            i = j
            if acc >= dist:
                break
        return self.centers[i], i

    def action(self, k, in_race=True):
        loc = np.asarray(k["location"], np.float64)
        front = np.asarray(k["front"], np.float64)
        speed = float(k["speed"])
        h = front - loc
        la = min(self.la_max, self.la0 + self.la_k * speed)
        tgt, ti = self.target(int(k["node"]), la)
        d = tgt - loc
        err = wrap(math.atan2(d[0], d[2]) - math.atan2(h[0], h[2]))  # >0: objetivo hacia +X (derecha)
        wb = float(k.get("wheel_base", 1.0))
        delta = math.atan2(2.0 * wb * math.sin(err), max(la, 1.0))
        steer = float(np.clip(self.sign * self.gain * delta / max(float(k["max_steer_angle"]), 1e-3), -1.0, 1.0))
        # curvatura adelante: angulo entre el rumbo al objetivo cercano y al lejano
        far, _ = self.target(int(k["node"]), la * 2.2)
        e2 = wrap(math.atan2(far[0] - loc[0], far[2] - loc[2]) - math.atan2(d[0], d[2]))
        curve = abs(err) + abs(e2)
        brake = bool(speed > self.brake_speed and curve > self.brake_curve)
        nitro = bool(float(k.get("energy", 0.0)) > 0.5 and abs(err) < self.nitro_err and curve < self.nitro_curve)
        # rescate: pulso corto (3 pasos) tras ~3 s sin moverse y luego ~5 s de espera; mantenerlo pulsado deja al kart en bucle
        rescue = False
        if self.cool > 0:
            self.cool -= 1
        if self.pulse > 0:
            self.pulse -= 1
            rescue = True
        elif in_race and speed < 0.5 and self.cool == 0:
            self.stuck += 1
            if self.stuck > 90:
                self.pulse, self.cool, self.stuck = 3, 150, 0
                rescue = True
        elif speed >= 0.5:
            self.stuck = 0
        drift = bool(self.drift and not brake and speed > self.drift_speed and abs(steer) > 0.55 and curve > self.drift_curve)
        return {"steer": steer, "acceleration": 0.0 if brake else 1.0, "brake": brake, "nitro": nitro,
                "drift": drift, "fire": False, "rescue": rescue}


def run(track, laps, sign, seed, max_seconds, verbose=True, **tk):
    from stk_agent.config import RaceSettings
    from stk_agent.gym_env import STKPixelEnv
    cfg = RaceSettings(track=track, kart="tux", laps=laps, num_karts=8, seed=seed, player_slot=0,
                       max_seconds=max_seconds, difficulty=2)
    env = STKPixelEnv(cfg, privileged=True)
    obs, info = env.reset(seed=seed)
    teacher = Teacher(env.base.track, steer_sign=sign, **tk)
    t0, steps, last = time.time(), 0, None
    done = False
    while not done:
        st = info["privileged"]
        k = st["karts"][0]
        a = teacher.action(k, in_race=st["phase"] in ("GO_PHASE", "MUSIC_PHASE", "RACE_PHASE"))
        act = {"steer": np.array([a["steer"]], np.float32), "acceleration": np.array([a["acceleration"]], np.float32),
               **{n: int(a[n]) for n in ("brake", "nitro", "drift", "fire", "rescue")}}
        obs, r, term, trunc, info = env.step(act)
        steps += 1
        done = term or trunc
        last = info["privileged"]["karts"][0]
    env.close()
    res = {"track": track, "finished": bool(last["has_finished_race"]), "finish_time": float(last["finish_time"]),
           "distance": float(last["overall_distance"]), "position": int(last["position"]), "steps": steps,
           "wall_s": round(time.time() - t0, 1)}
    if verbose:
        print(res, flush=True)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", nargs="+", default=["lighthouse"])
    p.add_argument("--laps", type=int, default=1)
    p.add_argument("--sign", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--max-seconds", type=float, default=150.0)
    p.add_argument("--drift", action="store_true")
    p.add_argument("--drift-curve", type=float, default=0.4)
    p.add_argument("--drift-speed", type=float, default=12.0)
    p.add_argument("--brake-curve", type=float, default=0.9)
    p.add_argument("--brake-speed", type=float, default=14.0)
    p.add_argument("--nitro-err", type=float, default=0.08)
    p.add_argument("--nitro-curve", type=float, default=0.25)
    a = p.parse_args()
    tk = dict(drift=a.drift, drift_curve=a.drift_curve, drift_speed=a.drift_speed, brake_curve=a.brake_curve, brake_speed=a.brake_speed,
              nitro_err=a.nitro_err, nitro_curve=a.nitro_curve)
    for t in a.tracks:
        run(t, a.laps, a.sign, a.seed, a.max_seconds, **tk)
