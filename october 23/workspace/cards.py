"""The cards of the editor page as plain HTML strings (inline styles, for viser's `add_html`) and the checklist. No viser import: tested as strings."""

import numpy as np

import editor as ed
import openarm_poses as poses

INK, GREY, GREEN, RED, AMBER, CARD_BG, BORDER = "#222", "#777", "#1f9d55", "#d62828", "#e09f00", "#f7f8fa", "#d5d9e0"
FONT = "font-family:system-ui,sans-serif;"


def _card(inner: str, accent: str = BORDER) -> str:
    return f'<div style="{FONT}background:{CARD_BG};border:1px solid {BORDER};border-left:6px solid {accent};border-radius:10px;padding:10px 12px;margin:4px 0;color:{INK};">{inner}</div>'


def _dot(color: str) -> str:
    return f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:{color};margin-right:6px;"></span>'


def position_card(side: str, joints_deg, gripper, hand_cm, fresh: bool) -> str:
    """One arm's live position: seven joint angles (degrees), the gripper, and the hand's x, y, z in cm (the arm's base frame).

    @param side `"left"` or `"right"`.
    @param joints_deg Seven angles, or None when there is no data.
    @param gripper The gripper's logical angle in degrees, or None.
    @param hand_cm `(x, y, z)` in cm, or None.
    @param fresh Whether the data is arriving now.
    """
    title = f'<div style="font-size:15px;font-weight:700;">{side.upper()} ARM</div>'
    if not fresh or joints_deg is None:
        return _card(f'{title}<div style="color:{GREY};margin-top:4px;">{_dot(AMBER)}waiting for the teleop... (start your teleop in the other terminal)</div>', AMBER)
    cells = "".join(
        f'<td style="text-align:center;padding:2px 5px;"><div style="color:{GREY};font-size:11px;">J{i + 1}</div><div style="font-size:15px;font-weight:600;">{v:.1f}</div></td>'
        for i, v in enumerate(joints_deg))
    grip = "" if gripper is None else f'<div style="margin-top:4px;color:{GREY};">gripper {gripper:.1f}&deg;</div>'
    hand = "" if hand_cm is None else (f'<div style="margin-top:6px;font-size:14px;"><b>hand</b> &nbsp; x {hand_cm[0]:.1f} &nbsp; y {hand_cm[1]:.1f} &nbsp; z {hand_cm[2]:.1f} cm</div>')
    return _card(f'{title}<div style="color:{GREY};font-size:12px;">{_dot(GREEN)}receiving from the teleop &middot; joint angles in degrees</div>'
                 f'<table style="margin-top:6px;border-collapse:collapse;"><tr>{cells}</tr></table>{grip}{hand}', GREEN)


def status_strip(receiving: dict[str, bool], clash: str | None) -> str:
    """The thin strip at the top: a light per arm and one line about clashes."""
    lights = "".join(f'{_dot(GREEN if on else AMBER)}{side} {"receiving" if on else "waiting"} &nbsp; ' for side, on in receiving.items())
    verdict = (f'<span style="color:{GREEN};font-weight:600;">No clash</span>' if clash is None
               else f'<span style="color:{RED};font-weight:600;">CLASH:</span> {clash}')
    return f'<div style="{FONT}font-size:12px;padding:4px 2px;">{lights}<br>{verdict}</div>'


def home_card(side: str, home_set: bool, has_path: bool, home_deg=None) -> str:
    """One arm's home: whether it is set, whether a path from rest is recorded, and the angles."""
    title = f'<div style="font-size:15px;font-weight:700;">{side.upper()} HOME</div>'
    if not home_set:
        return _card(f'{title}<div style="color:{RED};margin-top:4px;">home not set yet</div>', RED)
    angles = "" if home_deg is None else f'<div style="color:{GREY};font-size:12px;margin-top:4px;">{", ".join(f"{v:.1f}" for v in home_deg)} &deg;</div>'
    state = (f'<span style="color:{GREEN};">home set, path from rest recorded</span>' if has_path
             else f'<span style="color:{RED};">home set but NO path from rest: move the leader from the rest or ready pose, then press the button again</span>')
    return _card(f'{title}<div style="margin-top:4px;">{state}</div>{angles}', GREEN if has_path else RED)


def checklist(items: list[tuple[str, bool, str]]) -> str:
    """A tick or a cross per item, with the hint for a missing one."""
    rows = "".join(
        f'<div style="margin:3px 0;"><span style="color:{GREEN if ok else RED};font-weight:700;">{"&#10004;" if ok else "&#10008;"}</span> {label}'
        + ("" if ok or not hint else f'<div style="color:{GREY};font-size:12px;margin-left:20px;">{hint}</div>') + "</div>" for label, ok, hint in items)
    return _card(f'<div style="font-size:15px;font-weight:700;margin-bottom:4px;">Before you save</div>{rows}')


def build_checklist(state: ed.EditorState, session) -> list[tuple[str, bool, str]]:
    """What is done and what is missing, from the editor state and the teach session."""
    items: list[tuple[str, bool, str]] = []
    for side in ed.SIDES:
        arm, cap = state.arms[side], side.capitalize()
        built_in = arm.home_source == "ready" and not arm.home_path
        items.append((f"{cap} home set", arm.home_source != "default", f"Home tab: press Set {side.upper()} home here"))
        items.append((f"{cap} path recorded", bool(arm.home_path) or built_in or arm.home_source == "rest",
                      f"move the {side} leader from the rest or ready pose to the home, then press Set {side.upper()} home here"))
        items.append((f"{cap} allowed area", bool(arm.region), "Danger area tab: Record extremes and move the leader to the farthest safe points"))
    items.append(("Table placed", bool(state.table_set), "Danger area tab: enter the table numbers or drag the table"))
    items.append(("No clash", not state.clashes(), "a zone blocks the rest, ready or home pose: move or resize it"))
    return items
