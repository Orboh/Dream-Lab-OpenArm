# Work log index — october 23

| File | Why this log exists |
|------|---------------------|
| [log-001.md](log-001.md) | Generated the OpenArm v10 bimanual URDF with xacro from `openarm_description` (includes step-by-step procedure), so Dream-1 can use the real robot description. |
| [log-002.md](log-002.md) | Packed the generated URDF into `openarm_bimanual_urdf.zip` and `openarm_bimanual_urdf_with_meshes.zip` for sharing/upload. |
| [log-003.md](log-003.md) | Read the setup wizard's Driver step and the Dream driver template to decide how to drive one OpenArm arm safely (no code written yet). |
| [log-004.md](log-004.md) | Read Dream docs through the MCP server and researched what a full OpenArm setup needs (driver, built-in OpenArm model, mirror and sign questions). Nothing run on the arms. |
| [log-005.md](log-005.md) | Wrote the OpenArm driver (read-only and full classes) for the Dream setup wizard, tested on a fake CAN bus only. |
| [log-006.md](log-006.md) | Read-only probe of the follower motors after CAN bring-up: all 16 answer over CAN-FD, all disabled; three odd joint angles to check. |
| [log-007.md](log-007.md) | Why the wizard's Find arms step reports "Not found yet": only the right follower is being polled. |
| [log-008.md](log-008.md) | Driver now accepts `left` / `right` as arm addresses to get around the wizard merging the two follower links. |
| [log-009.md](log-009.md) | Wrote and validated the hand-made station.yaml for the OpenArm bimanual (dry-run driver, built-in OpenArm v2 model). |
| [log-010.md](log-010.md) | Checked whether the driver's joint mapping matches Dream's OpenArm v2 model before moving the arms: wrist (joint 5) is off by 90 deg; not safe to move yet. |
| [log-011.md](log-011.md) | Implemented "swing back then rise to the ready pose" at station start and parking at rest on stop (tested on the fake bus only). |
| [log-012.md](log-012.md) | Daily safety area part 1: zones schema and FK, hard driver guard, one-arm lock (`arms.py`), tests only. |
| [log-012.md](log-012.md) | Daily safety area: zones, guard, one-arm lock, capture/verify/build/apply tool and 3D viewer (tests only; viewer demo running read-only). |
| [log-013.md](log-013.md) | Round 2: real OpenArm model in the browser, visual zone editor, fixed output format (home pose + allowed/forbidden coordinates, station keys, layout picture). |
| [log-014.md](log-014.md) | The editor page as four cards (Position, Home, Danger area, Save) with automatic path recording and live allowed region. |
