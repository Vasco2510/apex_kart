"""Sonda: tipos y posiciones de items del estado privilegiado. python tools/probe_items.py <pista>"""
import sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))


def main():
    from teacher import Teacher
    from stk_agent.config import RaceSettings
    from stk_agent.gym_env import STKPixelEnv
    cfg = RaceSettings(track=sys.argv[1], kart="tux", laps=1, num_karts=8, seed=11, player_slot=0, max_seconds=60, difficulty=2)
    env = STKPixelEnv(cfg, privileged=True); obs, info = env.reset(seed=11)
    t = Teacher(env.base.track); types = collections.Counter(); ex = {}
    for i in range(1500):
        st = info["privileged"]; k = st["karts"][0]
        if i == 0:
            print("kart keys", sorted(k.keys())); print("n items", len(st["items"]), st["items"][:3])
        for it in st["items"]:
            types[it["type"]] += 1; ex[it["type"]] = it
        a = t.action(k, in_race=st["phase"] in ("GO_PHASE", "MUSIC_PHASE", "RACE_PHASE"))
        act = {"steer": np.array([a["steer"]], np.float32), "acceleration": np.array([a["acceleration"]], np.float32),
               **{n: int(a[n]) for n in ("brake", "nitro", "drift", "fire", "rescue")}}
        obs, r, te, tr, info = env.step(act)
        if te or tr: break
    print("TYPES", types); print("EX", ex)
    k = info["privileged"]["karts"][0]
    print("KART", {x: k[x] for x in k if x in ("powerup", "attachment", "energy", "max_steer_angle", "wheel_base", "speed")})
    env.close()


if __name__ == "__main__":
    main()
