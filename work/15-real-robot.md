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
5b. **Perception + calibration** ([16](16-perception.md), [17](17-calibration.md)): `camera_probe_web.py` shows the block box and base-frame xyz;
   `scripts/calibrate_camera.py from-demos --save` after a recording (re-run whenever the chest camera is touched).
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

## Using it: the web console
`orboh --backend bridge serve --host 127.0.0.1` (bridge fake or real running) → <http://127.0.0.1:8000>. Top card: type the task,
**Dry-run**, read the chunk verdicts, then approve **Run on robot** ([11](11-web-console.md)). With the real bridge the arms move.
Stop the console (Ctrl-C) before recording demos: it holds the cameras.

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
| Every plan rejected "at t=0.00s", joint values frozen, bridge log full of `drops [joint_1 21/21 … gripper 21/21]` | **the motors have stopped answering** (powered off, e-stop, CAN cable): `ip -details link show can_fol_l` says `ERROR-PASSIVE`, `candump can_fol_l` shows no frames, TX packets keep growing while RX stand still. The fork then returns the **last known** joint values, so the stale numbers made every plan fail the joint-limit check. The bridge now refuses to read or command while a bus is not `ERROR-ACTIVE` (takes effect on its next start). Fix the power / cable, support the arms (they are limp), then restart the bridge (it re-enables the motors and moves the arms to the start pose). Happened 2026-10-20 20:52 on both arms at once |
| Red LED on a motor | a Damiao motor reports an error code (or just sits disabled). Disabled motors only report "disabled", so passively listening shows nothing. Read the real code: with teleop/record/bridge stopped, `~/projects/lerobot/.venv/bin/python scripts/damiao_probe_state.py can_fol_l joint_6 --repeat 1` (in `openarm-orboh`; enable once at zero torque, read, disable). `state=0x1 enabled` = no latched error; `0x8`–`0xE` = the error type, then `damiao_errcheck.py <bus> --clear-motor <bus>:<joint>`, else power-cycle the arm (off, 10 s, on). Check the cable first (the 2026-10-20 J6 case was an unplugged cable). There is no software "lock" to release. 2026-10-20 check: all 16 follower motors `enabled`, 27–33 °C, no error |
| `bridge at … did not answer` | bridge not running / wrong endpoint; `ZmqRobot` closes the stuck socket, call `connect()` again |
| Every dry-run chunk fails `sim_collisions` at t≈0.2 s | bridge started from the fork's default pose; pass the pose files |
| Camera feed blank in the console | see [04 → No camera feed](04-cameras.md#no-camera-feed-2026-10-20-incident): missing camera (USB cable), or a wedged depth stream (`scripts/reset_realsense.py`) |
| `no camera for Dream slot(s)` | cameras not opened: pin serials, or another process (viewer, `lerobot-record`) holds them |

## Related
- [03 Robot layer](03-robot-layer.md) · [04 Cameras](04-cameras.md) · [14 Policy](14-policy.md) · [13 Roadmap](13-roadmap.md)
