[← instruction.md](instruction.md)

# 16 · Perception: finding the block (RealSense depth)

**Code:** `orboh/perception/{types,block_finder}.py`, `RealSenseSource.read_rgbd` in `orboh/cameras/sources.py`,
`scripts/camera_probe_web.py` (live overlay). **Tests:** `tests/test_perception.py`. Part of the pick pipeline
(find the block → grasp plan → safety check): the DreamLab/Dream-1 API cannot take an object position, so this runs on our side
([14](14-policy.md)).

## What it does
1. `RealSenseSource(..., depth=True)` streams depth aligned to colour (`rs.align`), in metres, with the colour intrinsics.
   Which roles stream depth: `cameras.depth_roles` (default none; `[center]` in `pipeline.yaml`, the D435i). The D405 wrists stay colour-only.
2. `BlockFinder.find(rgbd)` (classical, no model):
   - deprojects depth to camera-frame points inside `perception.roi` (the carpet; **not** the clear box, which gives bad depth);
   - RANSAC plane through the ROI (the table dominates it), refined by SVD; rejects the frame if < 40% of points fit a plane;
   - keeps what stands `min_height_mm`…`max_height_mm` above the plane, connected components, drops blobs with no depth;
   - per blob: height above the table, footprint (min-area rectangle in the table plane), top-face centre, object centre (half a
     height below the top), yaw in the plane, and which known pieces fit within `match_tol_mm` (`cube`, `cylinder_standing`, `cylinder_lying`);
   - blobs outside sane sizes are **dropped, never guessed**; an empty list means "nothing found".
3. `Detection.cam_xyz` is in the **camera frame** (x right, y down, z forward, metres). `base_xyz` (robot base frame) is filled by
   `Pipeline.find_block` / the probe page when `config/camera_extrinsics.yaml` exists ([17](17-calibration.md), ~25 mm accurate for now).

## Live result (2026-10-20, D435i, wood cube on the carpet)

| Quantity | Measured | Truth |
|---|---|---|
| Detected | 20 of 20 frames, labelled `cube` | 1 cube |
| Height above table | 49.4 mm | 47 mm |
| Footprint | 53 × 52 mm | 47 × 47 mm (edge pixels add a few mm) |
| Position jitter | 0.1 mm (std, x/y/z) | — |
| Camera-frame position | (0.034, 0.098, 0.384) m | — |
| Base-frame position (provisional calibration) | (0.315, 0.007, 0.349) m | table 0.325 + half block 0.0235 ✓ |
| Plane fit | 93% inliers | — |

Accuracy of the **absolute** position still depends on calibration; relative repeatability is excellent.

## Run it
`.venv/bin/python scripts/camera_probe_web.py` → <http://127.0.0.1:8765>: the overhead view has the box, size and camera xyz drawn on
the block; `/detect` returns JSON. Dream always gets the clean (undrawn) frame. Stop other programs that hold the cameras first.

## Config (`perception:` in `pipeline.yaml`)
`roi` (x0,y0,x1,y1 colour pixels, **re-check if the camera is moved**), `min_height_mm`, `max_height_mm`, `plane_tol_mm`,
`min_pixels`, `footprint_range_mm`, `height_range_mm`, `match_tol_mm`, `pieces`.

## Lessons
- Closing the mask morphologically fills pixels that have **no depth**; their point is (0,0,0), which stretched a 47 mm block to a 232 mm
  footprint and got it rejected. Blobs are now intersected with valid-depth pixels (regression test included).
- Colour thresholding alone is unreliable (pale wood on dark carpet, auto-exposure); depth above the table plane is not.

## Limits
Level 1 only (one object on plain carpet). Other materials, occlusion and clutter need a learned detector behind the same
`find(rgbd) -> list[Detection]` interface. Piece sizes in `perception.pieces` are the three wood pieces of the Kitakyushu scene
(the prism's lying footprint is not listed).

## Related
- [04 Cameras](04-cameras.md) · [14 Policy](14-policy.md) · [18 Grasp planner](18-grasp-planner.md) · [12 Testing](12-testing.md)
