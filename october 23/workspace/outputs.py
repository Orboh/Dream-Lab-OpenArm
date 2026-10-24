"""The editor's two outputs: the zones bundle with its layout picture, and the station keys. Both follow the fixed output format (see README)."""

import datetime as dt
import pathlib

import yaml

import editor as ed
import layout
import zonebuild as zb

HEADER = """# Station keys written by workspace.py (the safety area).
# Per arm: home = where the arm starts and ends a run (7 joint angles, radians);
#          reach = where the hand may be, seen from above ([x, y], metres, arm base frame);
#          keep_out = boxes the arm must stay out of ([x, y, z] centre and size, metres, arm base frame).
# Paste each block into ~/.dreamlabs/station.yaml, or run `workspace.py apply` which does it with a backup.
"""


def save_bundle(state: ed.EditorState, operator: str, directory: pathlib.Path, now: dt.datetime | None = None) -> tuple[pathlib.Path, pathlib.Path]:
    """Writes `zones-YYYYMMDD-HHMM.yaml` and `layout-YYYYMMDD-HHMM.png` into `directory`.

    @param state The editor's state.
    @param operator Who made it.
    @param directory Where to write (created).
    @param now The time; the system clock by default.
    @return `(yaml_path, png_path)`.
    """
    now = now or dt.datetime.now().astimezone()
    raw = state.to_bundle(operator, now)
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = f"{now:%Y%m%d-%H%M}"
    yaml_path = directory / f"zones-{stamp}.yaml"
    yaml_path.write_text(yaml.safe_dump(raw, sort_keys=False, default_flow_style=None))
    return yaml_path, layout.render_layout(raw, directory / f"layout-{stamp}.png")


def export_station_keys(raw: dict, path: pathlib.Path) -> pathlib.Path:
    """Writes the per-arm station keys (`home`, `reach`, `keep_out`) of a bundle, ready to paste or apply.

    @param raw The zones bundle.
    @param path The file to write.
    @return `path`.
    """
    homes = {side: (0.0, 0.0) for side in zb.SIDES}  # only the home cell of `reach` needs a hand position; the bundle's own home is used below
    import numpy as np

    import openarm_zones as zones

    homes = {side: tuple(zones.ArmKinematics(side, logical=True).points(np.asarray(raw["home"][side], dtype=float))["tcp"][:2]) for side in zb.SIDES}
    fragment = zb.station_fragment(raw, homes)
    keys = {"follower_l": fragment["left"], "follower_r": fragment["right"]}
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + yaml.safe_dump(keys, sort_keys=False, default_flow_style=None))
    return path
