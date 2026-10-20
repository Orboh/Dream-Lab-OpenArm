[← instruction.md](instruction.md)

# 04 · Cameras

**Code:** `orboh/cameras/sources.py`, `orboh/cameras/discovery.py`, with orchestration in `orboh/pipeline.py`.

## Expected hardware (Kitakyushu rig)
- **center:** D435i `344422070898` on a homemade chest mount (~700 mm above the plate, 45° tilt); also streams depth (`cameras.depth_roles`)
- **left_wrist:** D405 `409122272149`; **right_wrist:** D405 `352122273417` (colour only)

## Sources (`sources.py`)

| Class | Use |
|---|---|
| `list_realsense()` | enumerate devices: serial, name, USB type, physical port, firmware |
| `RealSenseSource` | color stream of one device (`rgb8`, configured resolution/fps); with `depth=True` also depth aligned to colour (`read_rgbd()` → `RGBD`, see [16](16-perception.md)) |
| `SimCameraSource` | MuJoCo render; in sim, `camera_wrist_right` and `tablecam` stand in for the real cameras |
| `CameraManager` | holds open sources **by role**, caches the latest JPEG for the UI |

## Discovery: which camera is which (`discovery.py`)

`assign_roles()` uses the strongest evidence available:

1. **Pinned serials** (`cameras.pinned_serials`): certain; discovery is skipped.
2. **Wiggle test** (optional, moves hardware):
   - The right gripper opens/closes by `motion_amplitude`.
   - Each camera scores the fraction of its view that changed, minus the frame-to-frame noise
     measured with nothing moving.
   - The fingers fill a large part of the wrist view but only a few pixels of the center view.
   - The top score → `right_wrist`.
   - Counts as confident only if the top score ≥ 0.25 and ≥ 2× the runner-up.
   - The motion goes through the full safety checks ([08](08-safety.md)).
3. **Model heuristic:** a D405 (short-range model) is probably the wrist camera. Never counted as
   confident on its own.
4. **Unresolved:** a guess, flagged for manual assignment.

The web console shows a live thumbnail per role. The operator can override and save
([11](11-web-console.md)).

**Why the gripper rather than an arm joint?** An arm joint moves the whole view, but the gripper
moves with the camera and can hide the change in low-texture scenes. In sim this test gave
0.55 vs 0.03 with the gripper but ~0.01 vs 0.01 with an elbow move. It also doesn't move the arm at all.

## Output: `config/cameras.yaml` (git-ignored, per machine)

```yaml
method: wiggle_test
confident: true
cameras:
  center:      {serial: "…", name: "…", port: "…", width: 640, height: 480, fps: 30}
  right_wrist: {serial: "…", …}
lerobot_record_cameras_arg: '{center: {type: intelrealsense, serial_number_or_name: "…", …}, …}'
```

- The pipeline reloads this file on start.
- Paste `lerobot_record_cameras_arg` into `--robot.cameras=` for `lerobot-record` ([05](05-demos-recording.md)).
- After discovery, pin the serials in `pipeline.yaml` for fully deterministic behavior.

## Device ownership
A RealSense device can only be opened by one process. The pipeline therefore:
- closes all streams before discovery, then reopens them;
- closes streams before teleop recording and reopens them afterwards.

## Related
- [05 Demos & recording](05-demos-recording.md) · [06 DreamLab](06-dreamlab.md) (images sent with plan requests) · [10 Pipeline](10-pipeline.md)
