# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

Proyecto final del curso Deep Learning (UTEC, prof. Victor Flores Benites): entrenar un agente visual que conduzca en SuperTuxKart. La base es `stk_agent` (extraída de `MarioCar.zip`, ya descomprimido). `docs/ENUNCIADO.md` resume el enunciado del profesor (en español; los campos `[COMPLETAR]` faltan — no inventar su contenido). Grupos de 2 personas; se permite (y se recomienda) usar los datos de todos los grupos; "vale todo" en técnica. La entrega es un agente listo para que el profesor lo ejecute de forma independiente (sin formato oficial ni restricciones de inferencia).

This directory sits inside a larger personal notes archive (`Z:\documentacion\IA_CHATS`, see its `AGENTS.md`), but **this subproject is a real Python package**, unlike the parent archive. Docs/notes are in Spanish.

## Commands

No test suite exists (`pyproject.toml` points pytest at a nonexistent `tests/`; `pytest` is only an optional extra). Python 3.11 recommended.

```text
python -m pip install -e .
python -m stk_agent doctor --engine --config configs/test_collect.json   # verify real engine
python -m stk_agent assets --config configs/collect.json                  # real track/kart ids (authoritative)
python -m stk_agent collect --config configs/collect.json --keymap configs/keymap.yaml --group G --driver D --out data/local [--episodes N]
python -m stk_agent validate <episode> [--deep]
python -m stk_agent audit <episode_or_root> --out audit.json
python -m stk_agent inspect <episode> --frame 0 [--show] [--full-state]
python -m stk_agent play examples/agent.json --config configs/evaluation.json --out runs/x      # one model vs bots
python -m stk_agent race a/agent.json b/agent.json --out runs/y                                  # up to 4 models
python -m stk_agent league m1.json m2.json --tracks lighthouse ... --seeds 1 2 --out runs/z [--execute]
```

All CLI commands print JSON; a result with `"valid": false` exits 2. `--out` for `play`/`race` must be a **new** folder (`mkdir(exist_ok=False)`).

## Architecture

Entry: `cli.py` (argparse subcommands, lazy imports per command) → modules below.

- **Engine isolation** — `engine.py` runs `backend.PySTKBackend` (wrapper over PySuperTuxKart2 0.7.4) in a **separate `multiprocessing` process** and talks to it via a pipe (`reset`/`step`/`assets`/`stop`/`close`). Don't call the native engine in-process.
- **Config** — `config.py`: frozen `RaceSettings` dataclass (validates e.g. `difficulty ∈ {0,1,2}`, `tick_hz | 120`, `policy_hz | tick_hz`, even width/height, `num_karts ≤ 20`) and `Participant`. Configs in `configs/*.json` load into it. Defaults: 640×360, tick 60 Hz, policy/RGB 30 Hz.
- **Action/observation contract** — `types.py`: `Action` (steer ∈[-1,1], acceleration ∈[0,1], brake/nitro/drift/fire/rescue as bools) validates strictly and never clips silently. Policy observation is `uint8 [H,W,3]` RGB **including HUD**; privileged state (karts, items, track geometry) is available only in training/data, not to the policy at evaluation.
- **Agent plug-in protocol** (evaluation) — an agent is a folder with `agent.json` (`entry`, `class`, `config`, optional `python`) and a class with `__init__(config)`, `reset()`, `act(rgb) -> Action`, `close()` (see `examples/agent_template.py`). `agents.AgentWorker` launches it as a **subprocess** (`stk_agent.worker`) and exchanges length-prefixed JSON+bytes packets (`protocol.py`) over stdin/stdout. Agent code must **not print to stdout** (the worker redirects it to stderr, which goes to `agent_N.log`). `checkpoint` in config is resolved relative to the manifest. The agent's interpreter must also have `stk-agent` installed.
- **Evaluation** — `evaluation.py`: `run_race` (lock-step: engine waits for every agent's action; a timeout invalidates the race rather than substituting a bot) and `run_league` (schedule of heats/rotations; without `--execute` only writes `schedule.json`).
- **Training** — `env.py` has `MultiAgentEnv`/`TrainingEnv` (custom `reward_fn` via `RewardContext`, optional privileged info); `gym_env.py` wraps it as Gymnasium `STKPixelEnv` (Dict action space; obs RGB; privileged state only through `info`). No learning algorithms or pretrained models are included.
- **Human data** — `collection.py` + `input.py` (pygame UI, keyboard/gamepad via `configs/keymap.yaml|json`) → `recording.py` writes episodes into `*.partial` folders and finalizes them. Episode layout: `meta.json, rgb.mp4, data.h5, track.npz, checksums.json, READY.json`. `_legacy_validation.py` (schema `1.0.0`) / `audit.py` validate; only episodes with `READY.json` count. RGB is stored at `rgb_hz` while actions/state are stored at 60 Hz, so RGB index ≠ tick.
- **Datasets** — `dataset.py`: `Episode` (`transition(t)`, `iter_transitions`, `iter_sequences`, `iter_action_blocks`) and `DemonstrationDataset` (over a root of episodes; defaults to human-source episodes and control phases). Example: `examples/read_dataset.py <episode_dir>`. `transfer.py` handles index/publish/pack for sharing episodes.
- **Windows runtime** — `runtime.py` / `runtime-install` prepares the native Windows runtime if the upstream wheel needs it.

## Gotchas

- Version strings disagree: `pyproject.toml` says 0.6.7, `__init__.__version__` says 0.5.0, and `dist/` holds a 0.5.0 wheel. `src/` is the source of truth for installed code.
- The professor lists `difficulty` up to 3, but `RaceSettings` only accepts 0–2; real track/kart ids come from `assets`, not from the lists in `docs/ENUNCIADO.md`.
- `seed`/`player_slot` may be `"random"` for collection; use an integer seed for evaluation/doctor. Effective values are saved in `meta.json`.

## Project workflow (apex_kart)

Own code lives beside the course library: `agents/` (deliverable agents: `bc_cnn` = behaviour cloning, `const_accel` = baseline), `training/` (`train_bc.py`, `rewards.py`), `tools/` (preprocess, evaluate, audit, Drive helpers), `configs/train/`, `colab/` (generated notebook), `results/` (small tracked outputs), `docs/` (`PLAN_PROYECTO.md` = status per step, `AVANCE.md` = log, `COLAB.md` = how to train). Heavy stuff (`data/`, `runs/`, weights) is git-ignored.

- Python for the repo: venv `C:\Users\Yuri Abel\venvs\apex` (3.11; pygame does not build on 3.14). rclone remote `gmail` = the paid Google account; the Drive layout is documented in `tools/apex.py`.
- Cycle: `python tools/apex.py sync` -> run `colab/entrenar.ipynb` (one notebook; all logic is in `training/pipeline.py`, presets in `configs/train/<preset>.json`) -> `python tools/apex.py pull <run>` -> `python tools/apex.py eval <run>`.
- Preprocessed episodes: `frames.npy` (160x90, stride 2) + `labels.npz` + `info.json`; subset lists are `results/subset_*.txt` (paths may contain fullwidth characters; `tools/preprocess.py` normalises them).
- Update `docs/PLAN_PROYECTO.md` statuses and `docs/AVANCE.md` after each step. Commits: no Claude co-author trailer (user preference).
