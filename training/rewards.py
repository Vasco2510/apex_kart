"""F5B.1: recompensa de progreso para RL / evaluacion de politicas con estado privilegiado.

reward = d(overall_distance) - 0.05*fuera_de_pista - 0.01*freno
Usa solo estado del propio kart; el estado privilegiado NUNCA es entrada de la politica.
"""
from stk_agent.env import RewardContext, kart_state


def progress_reward(ctx: RewardContext) -> float:
    b = kart_state(ctx.before, ctx.slot)
    a = kart_state(ctx.after, ctx.slot)
    # overall_distance arranca en ~-880 antes de la salida y salta al empezar: se recorta el paso
    # a lo fisicamente posible (~1.3 m por paso de politica con nitro).
    r = max(-2.0, min(2.0, a["overall_distance"] - b["overall_distance"]))
    if not a.get("is_on_road", True):
        r -= 0.05
    if ctx.action.brake:
        r -= 0.01
    return float(r)
