[← instruction.md](instruction.md)

# 01 · Environment setup

Environment files at a glance:

| Thing | Path | Purpose | In git? |
|---|---|---|---|
| Python virtualenv | `.venv/` | interpreter + dependencies | no |
| Secrets file | `.env` | API keys, web token | **no**, never commit |
| Secrets template | `.env.example` | documents every variable | yes |
| Pinned libraries | `requirements.txt` | exact version of every library in `.venv` | yes |
| Robot extras | `requirements-robot.txt` | pyrealsense2 (+ note on the LeRobot fork) | yes |

## Laptop (development, sim only)

```bash
bash scripts/setup_env.sh        # creates .venv from requirements.txt, installs `orboh`, copies .env
source .venv/bin/activate
```

Requires Python ≥ 3.10. Off-screen rendering needs `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU).

## Robot box

The arms connect to a separate Linux box (probably the Jetson "Thor").

```bash
git clone <this repo> ~/Orboh && cd ~/Orboh
LEROBOT_DIR=~/projects/lerobot bash scripts/setup_env.sh --robot
```

`--robot` additionally installs:
- **pyrealsense2.** On x86 this comes from pip. On aarch64 (Jetson), build librealsense with
  `-DBUILD_PYTHON_BINDINGS=ON` and make it importable inside `.venv`.
- **The Orboh LeRobot fork** (branch `feat/bi-openarm-soft-move-to`), installed as editable from `$LEROBOT_DIR`.

Two interpreters are involved on the robot box: Orboh's `.venv` (Python 3.12, also holds the Dream client
via `requirements-dream.txt`, which needs 3.11–3.13) and the LeRobot fork's own venv (3.13) that runs the
robot bridge ([15](15-real-robot.md)). Python 3.10 cannot run the Dream client.

SocketCAN must be up before starting: `sudo bash ~/projects/openarm-orboh/scripts/canup.sh` creates the fixed
names `can_fol_l`, `can_fol_r` (followers) and `can_lead_l`, `can_lead_r` (leaders).

## `.env` variables

| Variable | Used by | Notes |
|---|---|---|
| `DREAMLAB_API_KEY` | [06 DreamLab](06-dreamlab.md) | required when `dreamlab.backend: http` |
| `DREAMLAB_BASE_URL` | [06 DreamLab](06-dreamlab.md) | e.g. `https://…/v1`, no trailing slash |
| `ORBOH_WEB_TOKEN` | [11 Web console](11-web-console.md) | required whenever the console listens on the network |
| `ORBOH_CONFIG` | [02 Configuration](02-configuration.md) | optional alternate config path |
| `MUJOCO_GL` | [07 Simulation](07-simulation.md) | `egl` / `osmesa` |

The loader (`orboh/config.py:load_config`) reads `.env` from the project root. Values already set in the shell take priority.

## Dependencies (reproducible install)

- **`pyproject.toml`** declares the *direct* dependencies (loose versions). Optional extras: `realsense`, `dev`.
- **`requirements.txt`** pins *every* library (direct and transitive) to the exact version tested.
  `setup_env.sh` installs from it, so every machine gets an identical `.venv`.
- **`requirements-robot.txt`** holds robot-box-only extras.
- **`requirements-dream.txt`** holds the Dream client (`bash scripts/setup_env.sh --dream`, heavy: jax, av, boto3).

`pyzmq` (robot bridge client) is in `pyproject.toml` and `requirements.txt`.
To run pytest on a box with ROS 2 sourced, use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` (ROS's pytest plugins break import).

Manual install, without the script:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
```

**Adding or upgrading a library:**
1. Add it to `pyproject.toml`.
2. `pip install` it in `.venv`.
3. Run the tests ([12](12-testing.md)).
4. Regenerate the pins: `.venv/bin/pip freeze --exclude-editable`. Keep the header comment in `requirements.txt`.
5. Commit both files together.

## Related
- [02 Configuration](02-configuration.md) · [12 Testing](12-testing.md) · [13 Roadmap](13-roadmap.md)
