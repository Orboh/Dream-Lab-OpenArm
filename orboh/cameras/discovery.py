"""Work out which camera is the center (head) camera and which is on the right wrist.

Order of evidence:
  1. pinned serials in config                    -> certain
  2. model heuristic (D405 is the wrist model)   -> likely
  3. wiggle test: open/close the right gripper; the fingers fill a large part of
     the wrist view but only a few pixels of the center view -> strong
The web UI shows thumbnails and lets the operator confirm or override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from orboh.cameras.sources import CameraDevice
from orboh.config import CamerasConfig

WRIST_MODELS = ("D405",)


@dataclass
class DiscoveryResult:
    devices: list[CameraDevice]
    assignments: dict[str, str]  # role -> serial
    method: str
    confident: bool
    notes: list[str] = field(default_factory=list)
    motion_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"devices": [d.to_dict() for d in self.devices], "assignments": self.assignments,
                "method": self.method, "confident": self.confident, "notes": self.notes,
                "motion_scores": self.motion_scores}


def changed_fraction(before: np.ndarray, after: np.ndarray, threshold: int = 25) -> float:
    """Fraction of pixels whose brightness changed noticeably."""
    a = before.astype(np.int16).mean(axis=2)
    b = after.astype(np.int16).mean(axis=2)
    return float((np.abs(a - b) > threshold).mean())


def assign_roles(devices: list[CameraDevice], cfg: CamerasConfig,
                 motion_scores: dict[str, float] | None = None) -> DiscoveryResult:
    roles = cfg.expected_roles
    serials = [d.serial for d in devices]
    notes: list[str] = []
    if len(devices) != len(roles):
        notes.append(f"expected {len(roles)} cameras ({', '.join(roles)}), found {len(devices)}")

    pinned = {r: s for r, s in cfg.pinned_serials.items() if s}
    if pinned:
        missing = [f"{r}={s}" for r, s in pinned.items() if s not in serials]
        if missing:
            notes.append("pinned cameras not connected: " + ", ".join(missing))
        found = {r: s for r, s in pinned.items() if s in serials}
        return DiscoveryResult(devices, found, "pinned", not missing and len(found) == len(roles), notes)

    if motion_scores and len(motion_scores) >= 2:
        ranked = sorted(motion_scores.items(), key=lambda kv: kv[1], reverse=True)
        (wrist, top), (other, second) = ranked[0], ranked[1]
        confident = top >= 0.25 and top >= 2.0 * max(second, 0.01)
        if not confident:
            notes.append(f"wiggle test ambiguous (scores {top:.2f} vs {second:.2f}); please confirm")
        return DiscoveryResult(devices, {"right_wrist": wrist, "center": other}, "wiggle_test",
                               confident, notes, {k: round(v, 3) for k, v in motion_scores.items()})

    wrist = [d for d in devices if any(m in d.name for m in WRIST_MODELS)]
    if len(wrist) == 1 and len(devices) == 2:
        center = next(d for d in devices if d is not wrist[0])
        notes.append(f"{wrist[0].name} is a short-range wrist model; run the wiggle test to be sure")
        return DiscoveryResult(devices, {"right_wrist": wrist[0].serial, "center": center.serial},
                               "model_heuristic", False, notes)

    notes.append("cannot tell cameras apart by model; run the wiggle test or assign manually")
    guess = dict(zip(roles, serials))
    return DiscoveryResult(devices, guess, "unresolved", False, notes)


def save_cameras_yaml(result: DiscoveryResult, cfg: CamerasConfig, path: Path) -> dict:
    """Persist role mapping; also emits the --robot.cameras string for lerobot-record."""
    by_serial = {d.serial: d for d in result.devices}
    cams = {}
    for role, serial in result.assignments.items():
        d = by_serial.get(serial)
        cams[role] = {"serial": serial, "name": d.name if d else "", "port": d.port if d else "",
                      "width": cfg.width, "height": cfg.height, "fps": cfg.fps}
    lerobot_arg = "{" + ", ".join(
        f'{role}: {{type: intelrealsense, serial_number_or_name: "{c["serial"]}", '
        f'width: {c["width"]}, height: {c["height"]}, fps: {c["fps"]}}}'
        for role, c in cams.items()) + "}"
    doc = {"method": result.method, "confident": result.confident, "cameras": cams,
           "lerobot_record_cameras_arg": lerobot_arg}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return doc


def load_cameras_yaml(path: Path) -> dict[str, str]:
    """role -> serial from a saved discovery, or {}."""
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text()) or {}
    return {role: c["serial"] for role, c in (doc.get("cameras") or {}).items()}
