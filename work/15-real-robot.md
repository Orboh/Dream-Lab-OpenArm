[← instruction.md](instruction.md)

# 15 · Real robot via the bridge (new-setup guide)

**Code:** `scripts/robot_bridge.py`, `scripts/start_bridge.sh`, `orboh/robot/zmq_robot.py`,
`scripts/camera_probe_web.py`. **Tests:** `tests/test_zmq_robot.py`.

## Why a bridge

The real arms are driven by the team's LeRobot fork (`~/projects/lerobot`, branch
`feat/bi-openarm-soft-move-to`, Python 3.13 venv, SocketCAN + Damiao motors). The Dream client needs Python
3.11–3.13 and Orboh's `.venv` is 3.12. Rather than force everything into one interpreter, the arms live in
their own process and Orboh talks to it over ZMQ:

```
Orboh (.venv, py3.12)  --ZMQ REQ/REP, JSON-->  robot_bridge.py (lerobot venv)  --SocketCAN-->  arms
 ZmqRobot + JointMapping                          BiOpenArmFollower
```

The bridge speaks the fork's keys and **degrees** (`left_joint_1.pos` … `right_gripper.pos`); `JointMapping`
([03](03-robot-layer.md)) converts to canonical units on the Orboh side. Python 3.10 is not an option for the
Dream client (`dreamlabs` declares `Requires-Python >=3.11,<3.14`; a forced install fails on `scipy`).

## From scratch: checklist

1. **Repos.** Clone this repo, the LeRobot fork (`~/projects/lerobot`, branch above) and `openarm-orboh`
   (pose sets, scenes, CAN scripts). Install the fork into its own venv per its README.
2. **Orboh env.** `bash scripts/setup_env.sh --robot --dream` (PYTHON=python3.12), then edit `.env`
   (`DREAMLAB_API_KEY`, `DREAMLAB_BASE_URL`, `ORBOH_WEB_TOKEN`). ([01](01-environment-setup.md))
3. **CAN.** `sudo bash ~/projects/openarm-orboh/scripts/canup.sh`; follower buses are `can_fol_l`, `can_fol_r`
   (leader buses `can_lead_*` are not needed for autonomous runs). Check with `ip -br link | grep can`.
4. **Calibration.** Both followers must be calibrated (`bash scripts/calibrate_openarm.sh left|right` in
   `openarm-orboh`); the files live in `~/.cache/huggingface/lerobot/calibration/robots/`.
5. **Cameras.** Three RealSense devices; serials are pinned in `config/pipeline.yaml`
   (`cameras.pinned_serials`: D435i = `center`, D405 `409122272149` = `left_wrist`, D405 `352122273417` =
   `right_wrist`). On a different rig run `orboh discover-cameras --wiggle` and pin the result ([04](04-cameras.md)).
   Look at them: `.venv/bin/python scripts/camera_probe_web.py` → <http://127.0.0.1:8765>.
6. **Fake bridge + dry-run (no hardware moves).**
   ```bash
   bash scripts/start_bridge.sh fake        # terminal 1
   orboh --backend bridge preflight         # terminal 2: robot_connection, cameras, dream API
   orboh --backend bridge run-policy --policy-backend dream --dry-run --seconds 12 \
         --prompt "Pick up the wood and place it in the box"
   ```
   `start_bridge.sh` starts the fake arms from the pose set in `POSE_DIR`
   (default `openarm-orboh/config/pose_sets/kitakyushu-wood-l1-v1`).
7. **Real arms.** `bash scripts/start_bridge.sh real` (asks for confirmation) — **connecting makes both arms
   move to the start pose.** Then `orboh --backend bridge run-policy … --dry-run` once more (now with the real
   joint state), and only then `--approve` with a short `--seconds`, e-stop in reach.
   Stop the bridge with Ctrl-C: it disconnects and **releases torque, so the arms go limp** — support them first.

## Settings to check on a new rig (all in `config/pipeline.yaml`)

| Key | Value here | Check |
|---|---|---|
| `robot.backend` | `bridge` (or `--backend bridge`) | `sim` for laptop work |
| `robot.bridge.endpoint` / `timeout_ms` | `tcp://127.0.0.1:5560` / 2000 | match `start_bridge.sh` `ENDPOINT` |
| `robot.lerobot.joint_units` | `degrees` | the fork speaks degrees |
| `robot.lerobot.gripper_closed` / `gripper_open` | `0.0` / `-65.0` | fork limits are (−65, 0); direction confirmed from real demos (work/18) |
| `joint_signs` / `joint_offsets` | `{}` | **VERIFY** real vs MuJoCo direction ([07](07-simulation.md#sim-to-real-alignment)) |
| `cameras.pinned_serials` | three serials | per rig |
| `policy.camera_map` | overhead/left/right | wrong wrist assignment = wrong view for the model |

## Safety notes

- The Orboh safety stack ([08](08-safety.md)) still applies in full; the bridge adds the fork's own per-command
  clamp (`--max-relative-deg`, default 10°) as defence in depth.
- The bridge has **no watchdog of its own**: if Orboh dies mid-motion the arms hold their last target. Keep the
  e-stop in reach and Ctrl-C the bridge to release torque (arms go limp).
- Never skip the dry-run, and never `--approve` on a rig whose twin scene you have not compared with the table.

## Troubleshooting (from the team's 2026-10-20 report)

| Symptom | Cause / fix |
|---|---|
| CAN interfaces DOWN after reboot or replug | `sudo bash scripts/canup.sh` |
| `openarm_can` import error in motorcheck | apt package for system Python 3.10: run it with `/usr/bin/python3` (the bridge does not need it) |
| Follower J6 unresponsive, LED solid red | J6 cable unplugged; reseat it (enactic/openarm#471) |
| Wrist J7 spins ≈−80° when the hold releases | leader wrist falls under gravity (rest L 21.6°, R −15°); start-pose J7 must be that angle |
| Dataset unreadable after a recording | not finalized: always press esc and wait for shutdown before power-off (LeRobot v3 closes the parquet at finalize) |
| `bridge at … did not answer` | bridge not running / wrong endpoint; `ZmqRobot` closes the stuck socket, call `connect()` again |
| Every dry-run chunk fails `sim_collisions` at t≈0.2 s | bridge started from the fork's default pose; pass the pose files |
| `no camera for Dream slot(s)` | cameras not opened: pin serials, or another process (viewer, `lerobot-record`) holds them |

## Related
- [03 Robot layer](03-robot-layer.md) · [04 Cameras](04-cameras.md) · [14 Policy](14-policy.md) · [13 Roadmap](13-roadmap.md)
