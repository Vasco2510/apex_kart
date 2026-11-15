# Agente apex_kart (apex_kart_bc_r1)

Agente visual para SuperTuxKart: entra RGB 640x360 con HUD, sale `Action`. Sin acceso a estado privilegiado.
Red CNN (3 frames apilados, 160x90) entrenada por clonacion de comportamiento + DAgger (run `bc_teacher_r1`, pesos `model.pt`).
Incluye regla anti-atasco (si la imagen no cambia ~2 s, rescata).

## Uso
    pip install -e <repo>            # libreria stk_agent (+ torch, numpy)
    python -m stk_agent play agents/bc_cnn/final/agent.json --config configs/evaluation.json --out runs/prueba
    python -m stk_agent race agents/bc_cnn/final/agent.json otro/agent.json --out runs/carrera

`device` en agent.json: `cpu` por defecto (portable; ~12-20 ms/paso); `auto` usa GPU si existe.
Ajustes opcionales en `config`: `threshold` (0.5), `unstick` (true), `still_thr`, `rescue_after`.
