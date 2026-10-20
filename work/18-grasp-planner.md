[← instruction.md](instruction.md)

# 18 · Grasp planner (block → joint trajectory)

**Status:** gripper data measured from real demos (below); the planner itself (`orboh/planning/grasp.py`) is Phase 3 of the
perception plan and is not written yet. See [16 Perception](16-perception.md) for how the block position is obtained.

## Gripper values measured from the leader demos (2026-10-20)

Source: the 8 complete Kitakyushu wood-L1 gate episodes (`Orboh/bi_openarm_takeei_wood_l1_chest_wristr_wristl_v1_20261020_171837`,
5 left-arm, 3 right-arm picks): leader `action` vs follower `observation.state` for the picking arm's gripper, degrees.

| Quantity | Measured | Meaning |
|---|---|---|
| Closed | −0.6 … −1.7° (≈ 0) | `robot.lerobot.gripper_closed: 0.0` is right |
| Open | **negative**: −42 … −54° in practice (hard limit −65°) | `gripper_open: -65` points the right way: the open **direction is confirmed** |
| Follower while holding the wood | **−31.4 … −33.7°** (mean ≈ −32.2°, very consistent) | the angle at which the fingers meet the block; this is the contact angle |
| Leader command while holding | −14 … −26° | the operator squeezes past contact; the follower stalls at the block |

Canonical gripper value = `raw / −65` (0 closed … 1 open): contact −32° ≈ 0.49; typical open −48° ≈ 0.74; squeeze command −16° ≈ 0.25.

**Planner defaults taken from this** (config keys to be added in Phase 3): open to ≈ −48° before the descent, close to ≈ −16°
(command beyond contact so the position controller squeezes), expect the follower to stall near −32°; a stall far from that
means a miss (nothing, or the wrong thing, in the gripper).

**Width estimate (assumption, verify with calipers):** if the finger gap is roughly linear in angle and the 47 mm cube stalls at
−32°, then −48° ≈ 70 mm, i.e. about 11 mm of clearance per side around the cube. That is the tolerance budget for detection +
calibration error; keep both below ~8 mm.

## Related
- [16 Perception](16-perception.md) · [15 Real robot](15-real-robot.md) · [07 Simulation](07-simulation.md) · [13 Roadmap](13-roadmap.md)
