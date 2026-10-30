"""F5B.1: verifica STKPixelEnv + recompensa y mide pasos de politica por segundo.
Uso: python tools/check_env.py --steps 300
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import numpy as np
from rewards import progress_reward
from stk_agent.config import RaceSettings
from stk_agent.gym_env import STKPixelEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--track", default="lighthouse")
    a = p.parse_args()
    cfg = RaceSettings(track=a.track, kart="tux", laps=1, num_karts=4, seed=7, player_slot=0, max_seconds=120.0)
    env = STKPixelEnv(cfg, reward_fn=progress_reward, privileged=True)
    obs, info = env.reset(seed=7)
    print("obs", obs.shape, obs.dtype, "info keys", sorted(info)[:6])
    act = {"steer": np.zeros(1, np.float32), "acceleration": np.ones(1, np.float32),
           "brake": 0, "nitro": 0, "drift": 0, "fire": 0, "rescue": 0}
    total, t0 = 0.0, time.time()
    for i in range(a.steps):
        obs, r, term, trunc, info = env.step(act)
        total += r
        if term or trunc:
            break
    dt = time.time() - t0
    print(f"pasos={i + 1} recompensa_total={total:.2f} pasos_politica/s={(i + 1) / dt:.1f} term={term} trunc={trunc}")
    env.close()


if __name__ == "__main__":
    main()
