# AGENTS.md

## What this is
Course project for UTEC Deep Learning: train a **visual** agent (RGB + HUD only) to drive in SuperTuxKart, built on the `stk_agent` library extracted from `MarioCar.zip`. This is a real Python package (git repo `apex_kart`, branch `main`), distinct from the personal notes archive (`Z:\documentacion\IA_CHATS`) that contains it; don't apply that archive's doc-folder rules here. Docs and notes are in **Spanish**; keep new docs in Spanish.

`CLAUDE.md` has the deep architecture write-up. `docs/ENUNCIADO.md` is the professor's statement (fields marked `[COMPLETAR]` are unknown — do not invent). `docs/PLAN_PROYECTO.md` is the live phase plan; update its `Estado` column as work lands.

## Environment
- **Use Python 3.11** (README recommendation; the machine default is 3.14). `pyproject.toml` allows 3.10–3.14, but the engine wheel is validated on 3.11.
- `python -m pip install -e .` then `python -m stk_agent doctor --engine --config configs/test_collect.json` to verify the native engine.
- **No test suite**: `pyproject.toml` points pytest at a nonexistent `tests/`, and pytest is only an optional extra. Do not claim to run tests; verify with CLI commands.
- All CLI commands print JSON; a `"valid": false` result exits 2.

## Commands
```text
python -m stk_agent doctor --engine --config configs/test_collect.json
python -m stk_agent assets --config configs/collect.json          # authoritative track/kart ids
python -m stk_agent collect --config configs/collect.json --keymap configs/keymap.yaml --group G --driver D --out data/local [--episodes N]
python -m stk_agent validate <episode> [--deep]
python -m stk_agent audit <episode_or_root> --out audit.json
python -m stk_agent inspect <episode> --frame 0 [--show] [--full-state]
python -m stk_agent play agents/const_accel/agent.json --config configs/evaluation.json --out runs/x
python -m stk_agent race a/agent.json b/agent.json --out runs/y
python -m stk_agent league m1.json m2.json --tracks lighthouse --seeds 1 2 --out runs/z [--execute]
python tools/evaluate.py agents/const_accel/agent.json --set configs/eval_set.json --out runs/eval_x
```
- `tools/evaluate.py` is the F4.3 evaluation harness; it is **not** a `stk_agent` subcommand. It writes `results.csv` + `summary.json`.
- `--out` for `play`/`race`/`evaluate.py` must be a **new** folder (`mkdir(exist_ok=False)`).
- Use an integer `seed` for evaluation/doctor; `"random"` is only meaningful for collection.

## Architecture (non-obvious)
- `cli.py` (argparse, lazy imports) → modules. `engine.py` runs the native `PySTKBackend` in a **separate process** over a pipe; never call the engine in-process.
- `config.py`: frozen `RaceSettings` validates strictly. `difficulty` accepts **0–2 only**, despite the professor listing 3. Defaults 640×360, 60 Hz tick, 30 Hz policy/RGB.
- `types.py`: `Action` fields are `steer`, `acceleration`, `brake`, `nitro`, `drift`, `fire`, `rescue`; it validates and never silently clips.
- Agent protocol: folder with `agent.json` (`entry`, `class`, `config`, optional `python`) and a class with `__init__`, `reset`, `act(rgb)`, `close`. `agents.AgentWorker` runs it as a **subprocess**; agent code must **never print to stdout** (logs go to `agent_N.log`). `checkpoint` paths resolve relative to the manifest; the agent's interpreter also needs `stk-agent` installed.
- `evaluation.py` is lock-step: a slow/missing agent invalidates the race (no silent bot substitution).
- Data: episodes are `meta.json, rgb.mp4, data.h5, track.npz, checksums.json, READY.json`; only folders with `READY.json` count. RGB is stored at `rgb_hz`, actions/state at 60 Hz, so **RGB index ≠ tick**.
- `dataset.py` (`Episode`, `DemonstrationDataset`), `env.py`/`gym_env.py` (Gymnasium), `transfer.py` (sharing). No learning algorithms or pretrained models are included.

## Gotchas
- Version strings disagree: `pyproject.toml` 0.6.7, `__init__.__version__` 0.5.0, `dist/` 0.5.0 wheel. **`src/` is the source of truth.**
- Real track/kart ids come from `assets`, not from the lists in `docs/ENUNCIADO.md`.
- `data/`, `runs/`, `weights/`, `*.h5`, `*.mp4`, `*.npz`, `*.pt*` are gitignored on purpose; never commit datasets or checkpoints.
- `MarioCar.zip` is the course's original material; do not rename/delete it.
- Working tree may hold uncommitted experiment artifacts (`agents/`, `results/`, `tools/`, `configs/eval_*.json`, `runs/`); do not assume they are committed.
