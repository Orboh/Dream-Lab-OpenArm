"""Tests for the layout picture (top and side view): size, content and colours at known zone positions."""

import datetime as dt

import numpy as np
import pytest
from PIL import Image

import editor as ed
import layout

NOW = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)


def bundle():
    state = ed.EditorState.default()
    state.add_box("torso", (-0.15, 0.0, 0.35), (0.2, 0.5, 0.7), side="both")
    state.set_work_area("left", (0.40, 0.60), (0.0, 0.20))
    return state, state.to_bundle("t", NOW)


def near(pixel, color, tol=60):
    return all(abs(int(a) - int(b)) <= tol for a, b in zip(pixel[:3], color))


def test_render_writes_a_png_with_two_panels(tmp_path):
    _, raw = bundle()
    out = layout.render_layout(raw, tmp_path / "layout.png")
    img = Image.open(out)
    assert img.format == "PNG" and img.size == layout.SIZE
    assert len(img.convert("RGB").getcolors(maxcolors=1 << 20)) > 20  # not blank


def test_table_danger_box_and_work_area_are_drawn_where_they_are(tmp_path):
    state, raw = bundle()
    img = Image.open(layout.render_layout(raw, tmp_path / "l.png")).convert("RGB")
    t = state.table
    top_px = layout.project_top((t.x_range[0] + t.x_range[1]) / 2, 0.25)       # table, seen from above, away from the work-area dots
    assert near(img.getpixel(top_px), layout.BLUE_FILL)
    side_px = layout.project_side(-0.15, 0.35)                                   # centre of the torso box, side view
    assert near(img.getpixel(side_px), layout.RED_FILL)
    cell_world = ed.to_world("left", (0.5, 0.1, 0.0))  # a painted cell of the left arm (its base frame), drawn in the body frame
    assert near(img.getpixel(layout.project_top(cell_world[0], cell_world[1])), layout.GREEN, tol=30)
    # nothing is drawn where nothing is: far corner of the top view
    assert near(img.getpixel(layout.project_top(0.95, 0.55)), layout.BACKGROUND, tol=10)


def test_render_creates_the_folder_and_overwrites(tmp_path):
    _, raw = bundle()
    target = tmp_path / "deeper" / "layout.png"
    layout.render_layout(raw, target)
    layout.render_layout(raw, target)
    assert target.exists()


def test_region_box_is_drawn_in_green_in_both_views(tmp_path):
    state, _ = bundle()
    state.arms["left"].region = {"lo": [0.10, -0.20, -0.45], "hi": [0.60, 0.20, -0.05]}
    raw = state.to_bundle("t", NOW)
    img = Image.open(layout.render_layout(raw, tmp_path / "r.png")).convert("RGB")
    lo = ed.to_world("left", raw["arms"]["left"]["region"]["lo"])
    hi = ed.to_world("left", raw["arms"]["left"]["region"]["hi"])
    px, py = layout.project_side((lo[0] + hi[0]) / 2, hi[2])          # the middle of the top edge, side view
    assert any(near(img.getpixel((px, py + dy)), layout.GREEN, tol=40) for dy in range(-3, 4))
    px, py = layout.project_top(hi[0], (lo[1] + hi[1]) / 2)           # the middle of the far edge, top view
    assert any(near(img.getpixel((px, py + dy)), layout.GREEN, tol=40) for dy in range(-3, 4))
