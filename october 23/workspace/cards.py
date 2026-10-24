"""The cards of the editor page as plain HTML strings (inline styles, for viser's `add_html`) and the checklist. No viser import: tested as strings."""

import re

import numpy as np

import editor as ed
import i18n
import openarm_poses as poses

INK, GREY, GREEN, RED, AMBER, CARD_BG, BORDER = "#222", "#777", "#1f9d55", "#d62828", "#e09f00", "#f7f8fa", "#d5d9e0"
SCALE = 1.15  # card text relative to the page's own text, which `PAGE_CSS` makes larger and responsive
FONT = "font-family:system-ui,sans-serif;font-size:14px;"
# One global rule for the whole page (viser sizes everything in rem): the text grows with the window, from a readable 17 px to 20 px on big screens.
# The panel is clamped to the window by viser, so a phone or a small laptop still fits it.
PAGE_CSS = "<style>html{font-size:clamp(17px,0.35vw + 12px,20px);} .mantine-Button-label,.mantine-Tabs-tabLabel{white-space:normal;}</style>"


def _big(html: str) -> str:
    """Turns every `font-size:Npx` in a card's inline styles into rem, so the cards follow the page's responsive text size."""
    return re.sub(r"font-size:(\d+)px", lambda m: f"font-size:{int(m.group(1)) / 16 * SCALE:.3f}rem", html)


def _cap(lang: i18n.Lang, text: str) -> str:
    """English labels start with a capital; Japanese has none."""
    return text[:1].upper() + text[1:] if lang.code == "en" else text


def _card(inner: str, accent: str = BORDER) -> str:
    return _big(f'<div style="{FONT}background:{CARD_BG};border:1px solid {BORDER};border-left:6px solid {accent};border-radius:10px;padding:10px 12px;margin:4px 0;color:{INK};">{inner}</div>')


def _dot(color: str) -> str:
    return f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:{color};margin-right:6px;"></span>'


def position_card(side: str, joints_deg, gripper, hand_cm, fresh: bool, lang: i18n.Lang = i18n.EN, pitch_deg: float | None = None) -> str:
    """One arm's live position: seven joint angles (degrees), the gripper, and the hand's x, y, z in cm (the arm's base frame).

    @param side `"left"` or `"right"`.
    @param joints_deg Seven angles, or None when there is no data.
    @param gripper The gripper's logical angle in degrees, or None.
    @param hand_cm `(x, y, z)` in cm, or None.
    @param fresh Whether the data is arriving now.
    """
    name = lang.t(f"side.{side}")
    title = f'<div style="font-size:15px;font-weight:700;">{lang.t("pos.title", side=name.upper() if lang.code == "en" else name)}</div>'
    if not fresh or joints_deg is None:
        return _card(f'{title}<div style="color:{GREY};margin-top:4px;">{_dot(AMBER)}{lang.t("pos.waiting")}</div>', AMBER)
    cells = "".join(
        f'<td style="text-align:center;padding:2px 5px;"><div style="color:{GREY};font-size:11px;">J{i + 1}</div><div style="font-size:15px;font-weight:600;">{v:.1f}</div></td>'
        for i, v in enumerate(joints_deg))
    grip = "" if gripper is None else f'<div style="margin-top:4px;color:{GREY};">{lang.t("pos.gripper")} {gripper:.1f}&deg;</div>'
    hand = "" if hand_cm is None else (f'<div style="margin-top:6px;font-size:14px;"><b>{lang.t("pos.hand")}</b> &nbsp; x {hand_cm[0]:.1f} &nbsp; y {hand_cm[1]:.1f} &nbsp; z {hand_cm[2]:.1f} cm</div>')
    pitch = "" if pitch_deg is None else (f'<div style="margin-top:4px;font-size:14px;color:{GREEN if abs(pitch_deg) <= 15 else RED};"><b>{lang.t("pos.pitch", deg=pitch_deg)}</b></div>')
    return _card(f'{title}<div style="color:{GREY};font-size:12px;">{_dot(GREEN)}{lang.t("pos.receiving")}</div>'
                 f'<table style="margin-top:6px;border-collapse:collapse;"><tr>{cells}</tr></table>{grip}{hand}{pitch}', GREEN)


def status_strip(receiving: dict[str, bool], clash: str | None, lang: i18n.Lang = i18n.EN) -> str:
    """The thin strip at the top: a light per arm and one line about clashes."""
    lights = "".join(f'{_dot(GREEN if on else AMBER)}{lang.t(f"side.{side}")} {lang.t("strip.receiving" if on else "strip.waiting")} &nbsp; ' for side, on in receiving.items())
    verdict = (f'<span style="color:{GREEN};font-weight:600;">{lang.t("strip.noclash")}</span>' if clash is None
               else f'<span style="color:{RED};font-weight:600;">{lang.t("strip.clash")}</span> {clash}')
    return _big(f'<div style="{FONT}font-size:12px;padding:4px 2px;">{lights}<br>{verdict}</div>')


def home_card(side: str, home_set: bool, has_path: bool, home_deg=None, lang: i18n.Lang = i18n.EN) -> str:
    """One arm's home: whether it is set, whether a path from rest is recorded, and the angles."""
    name = lang.t(f"side.{side}")
    title = f'<div style="font-size:15px;font-weight:700;">{lang.t("home.title", side=name.upper() if lang.code == "en" else name)}</div>'
    if not home_set:
        return _card(f'{title}<div style="color:{RED};margin-top:4px;">{lang.t("home.unset")}</div>', RED)
    angles = "" if home_deg is None else f'<div style="color:{GREY};font-size:12px;margin-top:4px;">{", ".join(f"{v:.1f}" for v in home_deg)} &deg;</div>'
    state = (f'<span style="color:{GREEN};">{lang.t("home.ok")}</span>' if has_path
             else f'<span style="color:{RED};">{lang.t("home.nopath")}</span>')
    return _card(f'{title}<div style="margin-top:4px;">{state}</div>{angles}', GREEN if has_path else RED)


def checklist(items: list[tuple[str, bool, str]], lang: i18n.Lang = i18n.EN) -> str:
    """A tick or a cross per item, with the hint for a missing one."""
    rows = "".join(
        f'<div style="margin:3px 0;"><span style="color:{GREEN if ok else RED};font-weight:700;">{"&#10004;" if ok else "&#10008;"}</span> {label}'
        + ("" if ok or not hint else f'<div style="color:{GREY};font-size:12px;margin-left:20px;">{hint}</div>') + "</div>" for label, ok, hint in items)
    return _card(f'<div style="font-size:15px;font-weight:700;margin-bottom:4px;">{lang.t("check.title")}</div>{rows}')


def build_checklist(state: ed.EditorState, session, lang: i18n.Lang = i18n.EN) -> list[tuple[str, bool, str]]:
    """What is done and what is missing, from the editor state and the teach session."""
    items: list[tuple[str, bool, str]] = []
    for side in ed.SIDES:
        arm, name = state.arms[side], lang.t(f"side.{side}")
        built_in = arm.home_source == "ready" and not arm.home_path
        items.append((_cap(lang, lang.t("check.home", side=name)), arm.home_source != "default", lang.t("check.home.hint", side=name)))
        items.append((_cap(lang, lang.t("check.path", side=name)), bool(arm.home_path) or built_in or arm.home_source == "rest", lang.t("check.path.hint", side=name)))
        if state.danger_only:  # only the danger zones are named: no allowed area, work area or reference points are asked for
            continue
        items.append((_cap(lang, lang.t("check.area", side=name)), bool(arm.region), lang.t("check.area.hint")))
        over, total = state.reach_over_table(side)
        items.append((_cap(lang, lang.t("check.reachtable", side=name)), over > 0, lang.t("check.reachtable.hint", side=name, over=over, total=total)))
        holes = session.reach_holes(side)
        items.append((_cap(lang, lang.t("check.holes", side=name)), holes == 0, lang.t("check.holes.hint", n=holes)))
    for side in ([] if state.danger_only else ed.SIDES):
        name = lang.t(f"side.{side}")
        items.append((_cap(lang, lang.t("check.refs", side=name)), len(state.references[side]) >= 2, lang.t("check.refs.hint", side=name)))
    if not state.danger_only:
        overlap = state.region_overlap_cm()
        items.append((lang.t("check.overlap"), overlap == 0.0, lang.t("check.overlap.hint", cm=f"{overlap:g}")))
    items.append((lang.t("check.table"), bool(state.table_set), lang.t("check.table.hint")))
    items.append((lang.t("check.clash"), not state.clashes(), lang.t("check.clash.hint")))
    return items


def collect_card(snap: dict, lang: i18n.Lang = i18n.EN) -> str:
    """The collection card: what the run is doing now, the episode count, the run folder and the last console lines.

    @param snap From `CollectSession.snapshot()`.
    """
    import html

    phase = snap["phase"]
    color = {"recording": RED, "ready": GREEN, "idle": GREY, "finished": GREY}.get(phase, AMBER)
    state = lang.t("col.state.finished", rc=snap["rc"]) if phase == "finished" else lang.t(f"col.state.{phase}", n=snap["episode"])
    left = ""
    if snap.get("saved"):
        left += f'<div style="color:{GREEN};font-size:14px;font-weight:700;margin-top:4px;">&#10004; {lang.t("col.saved", n=snap["saved"])}</div>'
    if snap.get("target") is not None:
        left = (f'<div style="margin-top:6px;font-size:22px;font-weight:800;color:{RED if snap["left"] else GREEN};">{lang.t("col.left", left=snap["left"])}</div>'
                f'<div style="color:{GREY};font-size:13px;">{lang.t("col.left.detail", done=snap["done"], total=snap["target"])}</div>')
    folder = f'<div style="color:{GREY};font-size:12px;margin-top:4px;">{lang.t("col.run")}: {html.escape(snap["run_dir"])}</div>' if snap["run_dir"] else ""
    count = f'<div style="color:{GREY};font-size:12px;margin-top:4px;">{lang.t("col.episodes.done")}: {snap["started"]}</div>' if snap["started"] else ""
    tail = "\n".join(html.escape(line[:160]) for line in snap["lines"])
    log = (f'<div style="color:{GREY};font-size:11px;margin-top:6px;">{lang.t("col.log")}</div>'
           f'<pre style="margin:2px 0 0;font-size:11px;white-space:pre-wrap;max-height:180px;overflow:auto;">{tail}</pre>') if tail else ""
    return _card(f'<div style="font-size:15px;font-weight:700;">{_dot(color)}{state}</div>{left}{count}{folder}{log}', color)


# Keys for the Collect tab, as they were in the terminal. innerHTML never runs <script>, so the code rides on an image's onerror.
# It clicks the page's own buttons (found by the "(→)" marks in their labels), only while the Collect tab is showing and the button is enabled,
# and it takes the key before the tab bar or the 3D view can use it.
KEYS_HTML = (
    '<img src="x" style="display:none" onerror="if(!window.__collectKeys){window.__collectKeys=1;'
    "var M={ArrowRight:'(\u2192)',ArrowLeft:'(\u2190)',a:'(a)',A:'(a)',Escape:'(Esc)'};"
    "window.addEventListener('keydown',function(e){var m=M[e.key];if(!m||e.repeat||e.ctrlKey||e.metaKey||e.altKey)return;"
    "var t=e.target&&e.target.tagName;if(t==='INPUT'||t==='TEXTAREA'||t==='SELECT'||(e.target&&e.target.isContentEditable))return;"
    "var bs=document.querySelectorAll('button'),b=null;for(var i=0;i<bs.length;i++){if(bs[i].textContent.indexOf(m)>=0&&bs[i].offsetParent!==null){b=bs[i];break;}}"
    "if(!b)return;e.preventDefault();e.stopPropagation();if(!b.disabled&&b.getAttribute('data-disabled')===null)b.click();},true);}\">"
)
