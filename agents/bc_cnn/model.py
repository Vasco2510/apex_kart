"""Red CNN para clonacion de comportamiento (F5A). Compartida por entrenamiento y agente."""
import torch
from torch import nn

# Salidas: steer (regresion), acceleration, brake, nitro, drift, fire (logits). rescue se fuerza a 0.
N_OUT = 6


class Net(nn.Module):
    def __init__(self, frames=3, height=90, width=160, width_mult=1.0, dropout=0.2):
        super().__init__()
        c = [int(x * width_mult) for x in (32, 64, 96, 128)]
        self.conv = nn.Sequential(
            nn.Conv2d(3 * frames, c[0], 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(c[0], c[1], 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c[1], c[2], 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c[2], c[3], 3, 2, 1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 3 * frames, height, width)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, N_OUT))
        self.meta = dict(frames=frames, height=height, width=width, width_mult=width_mult, dropout=dropout)

    def forward(self, x):  # x: (B, 3*frames, H, W) float en [0, 1]
        return self.head(self.conv(x))


def decode(out, thr=0.5):
    """Logits (6,) -> dict de accion valido para stk_agent.Action."""
    steer = float(torch.tanh(out[0]))
    p = torch.sigmoid(out[1:])
    return {"steer": max(-1.0, min(1.0, steer)), "acceleration": float(p[0] > thr), "brake": bool(p[1] > thr),
            "nitro": bool(p[2] > thr), "drift": bool(p[3] > thr), "fire": bool(p[4] > thr), "rescue": False}
