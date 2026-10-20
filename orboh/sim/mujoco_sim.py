"""MuJoCo digital twin of the bimanual OpenArm (official enactic/openarm_mujoco model)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from orboh.types import GRIPPER_IDX, JOINT_NAMES, NUM_JOINTS, SIDES

FINGER_BODY_TAGS = ("ee_base_link", "inner_finger", "outer_finger")


def resolve_scene(scene: str) -> str:
    path = Path(scene).expanduser()
    if path.exists():
        return str(path)
    from openarm_mujoco.v2 import asset_path

    resolved = asset_path(scene)
    if not Path(resolved).exists():
        raise FileNotFoundError(f"MuJoCo scene not found: {scene}")
    return resolved


class MujocoSim:
    def __init__(self, scene: str, table_drop_m: float = 0.0, torso_tolerance_m: float = 0.0,
                 box: dict | None = None):
        self.scene = resolve_scene(scene)
        self._init_args = (scene, table_drop_m, box)
        if box:
            spec = mujoco.MjSpec.from_file(self.scene)
            self._add_box(spec, box, table_drop_m)
            self.model = spec.compile()
        else:
            self.model = mujoco.MjModel.from_xml_path(self.scene)
        self._object: tuple[np.ndarray, float] | None = None  # (xyz, yaw) of the manipulated object, applied on reset
        if table_drop_m:
            for body in ("table", "puck"):  # the puck rests on the table
                self.model.body(body).pos[2] -= table_drop_m
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self._renderer: mujoco.Renderer | None = None
        self._render_data: mujoco.MjData | None = None  # private copy of the state, so rendering never holds the lock
        # GL contexts are bound to the thread that created them: render on one thread only.
        self._render_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco-render")
        self._qadr = np.zeros(NUM_JOINTS, dtype=int)
        self._act = np.zeros(NUM_JOINTS, dtype=int)
        self._grip_lo = np.zeros(NUM_JOINTS)  # closed end in sim units
        self._grip_hi = np.zeros(NUM_JOINTS)  # open end in sim units
        for i, name in enumerate(JOINT_NAMES):
            side, part = name.split("_", 1)
            if part == "gripper":
                jname, aname = f"openarm_{side}_finger_joint1", f"{side}_finger1_ctrl"
            else:
                jname, aname = f"openarm_{side}_{part}", f"{side}_{part}_ctrl"
            self._qadr[i] = self.model.jnt_qposadr[self.model.joint(jname).id]
            self._act[i] = self.model.actuator(aname).id
            if part == "gripper":
                lo, hi = self.model.jnt_range[self.model.joint(jname).id]
                # Closed = the bound nearest zero; open = the far bound. VERIFY on hardware.
                self._grip_lo[i], self._grip_hi[i] = (lo, hi) if abs(lo) < abs(hi) else (hi, lo)
        self._robot_bodies = self._classify_bodies()
        self._torso_tol = torso_tolerance_m
        self._torso_pairs: list[tuple[int, int]] = []
        if torso_tolerance_m > 0:
            self._tolerate_torso()
        self.reset()

    @staticmethod
    def _add_box(spec, box: dict, table_drop_m: float) -> None:
        """An open-topped box on the table (floor + 4 walls): `center` [x, y], `size` [depth_x, width_y, height] in metres."""
        table_top = 0.40 - table_drop_m  # the pedestal scenes' table top, lowered like the table itself
        cx, cy = box["center"]
        dx, dy, h = box["size"]
        t = float(box.get("wall", 0.01))
        walls = {"floor": ((0, 0, t / 2), (dx / 2, dy / 2, t / 2)),
                 "wall_n": ((dx / 2, 0, h / 2), (t / 2, dy / 2, h / 2)), "wall_s": ((-dx / 2, 0, h / 2), (t / 2, dy / 2, h / 2)),
                 "wall_e": ((0, dy / 2, h / 2), (dx / 2, t / 2, h / 2)), "wall_w": ((0, -dy / 2, h / 2), (dx / 2, t / 2, h / 2))}
        for name, (off, half) in walls.items():
            spec.worldbody.add_geom(name=f"box_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half),
                                    pos=[cx + off[0], cy + off[1], table_top + off[2]], rgba=[0.8, 0.8, 0.9, 0.35],
                                    contype=1, conaffinity=1)

    def planning_copy(self) -> "MujocoSim":
        """The same scene with the torso kept in the contact pass: collision-aware IK needs it (the check sim relaxes it)."""
        scene, drop, box = self._init_args
        return MujocoSim(scene, drop, 0.0, box)

    @property
    def table_top(self) -> float:
        g = self.model.geom("table_top")
        return float(self.data.geom_xpos[g.id][2] + g.size[2])

    def set_object(self, xyz, half_extents=(0.0235, 0.0235, 0.0235), yaw: float = 0.0) -> None:
        """Put the manipulated object (the scene's free body, 'puck') at `xyz`, resized to a box of `half_extents`."""
        gid = self.model.geom("puck_geom").id
        self.model.geom_size[gid] = half_extents
        self._object = (np.asarray(xyz, float), float(yaw))
        self.reset(self.joints())

    def _place_object(self) -> None:
        if self._object is None:
            return
        body = self.model.body("puck")
        adr = int(self.model.jnt_qposadr[self.model.body_jntadr[body.id]])
        xyz, yaw = self._object
        self.data.qpos[adr:adr + 3] = xyz
        self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[:] = 0

    # ------------------------------------------------------------------ state
    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.zeros(NUM_JOINTS)
        hi = np.ones(NUM_JOINTS)
        for i, name in enumerate(JOINT_NAMES):
            if i in GRIPPER_IDX:
                continue
            side, part = name.split("_", 1)
            lo[i], hi[i] = self.model.jnt_range[self.model.joint(f"openarm_{side}_{part}").id]
        return lo, hi

    def home(self) -> np.ndarray:
        """Home pose from the model keyframe (elbows bent), else zeros."""
        if self.model.nkey:
            qpos = self.model.key_qpos[0]
            return self._qpos_to_canonical(qpos)
        return np.zeros(NUM_JOINTS)

    def reset(self, q: np.ndarray | None = None) -> None:
        with self.lock:
            if self.model.nkey:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            else:
                mujoco.mj_resetData(self.model, self.data)
            q = self.home() if q is None else np.asarray(q, dtype=float)
            sim_q = self._canonical_to_sim(q)
            for i in range(NUM_JOINTS):
                self.data.qpos[self._qadr[i]] = sim_q[i]
                if i in GRIPPER_IDX:  # second finger joint is coupled; mirror it
                    self.data.qpos[self._qadr[i] + 1] = sim_q[i]
            self.data.ctrl[self._act] = sim_q
            self._place_object()
            mujoco.mj_forward(self.model, self.data)

    def set_command(self, q: np.ndarray) -> None:
        with self.lock:
            self.data.ctrl[self._act] = self._canonical_to_sim(np.asarray(q, dtype=float))

    def step(self, seconds: float) -> None:
        with self.lock:
            for _ in range(max(1, int(round(seconds / self.model.opt.timestep)))):
                mujoco.mj_step(self.model, self.data)

    def joints(self) -> np.ndarray:
        with self.lock:
            return self._qpos_to_canonical(self.data.qpos)

    # --------------------------------------------------------------- contacts
    def contacts(self) -> list[tuple[str, str, str]]:
        """Contacts as (kind, body_a, body_b); kind in {ok, finger_env, self, arm_arm, env}."""
        out = []
        m, d = self.model, self.data
        with self.lock:
            for c in d.contact[: d.ncon]:
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                out.append((self._contact_kind(b1, b2), m.body(b1).name, m.body(b2).name))
            out += self._torso_overlaps()
        return out

    def _tolerate_torso(self) -> None:
        """Let the arms overlap the torso mesh by up to `torso_tolerance_m` and no further.

        The twin's simplified torso collision mesh is a centimetre or two stricter than the real robot, and its
        contact force would also stop the sim arm following a demo the real arm performed. So the torso geom is
        taken out of MuJoCo's contact pass and replaced by an explicit penetration test in `contacts()`.
        The arm's own mount (base link, link1) is not tested: it sits against the torso by construction.
        """
        m = self.model
        torso = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("openarm_body_link0_collision")]
        mount = {b for b in self._robot_bodies if m.body(b).name.endswith(("_base_link", "_link1"))}
        arm = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self._robot_bodies
               and int(m.geom_bodyid[g]) not in mount and m.geom_contype[g]]
        self._torso_pairs = [(t, a) for t in torso for a in arm]
        for t in torso:
            m.geom_contype[t] = m.geom_conaffinity[t] = 0

    def _torso_overlaps(self) -> list[tuple[str, str, str]]:
        m, d = self.model, self.data
        fromto = np.zeros(6)
        out = []
        for t, a in self._torso_pairs:
            if mujoco.mj_geomDistance(m, d, t, a, 0.0, fromto) < -self._torso_tol:
                out.append(("env", m.body(m.geom_bodyid[t]).name, m.body(m.geom_bodyid[a]).name))
        return out

    def _classify_bodies(self) -> dict[int, tuple[str, bool]]:
        """body id -> (side, is_finger) for robot bodies."""
        out = {}
        for b in range(self.model.nbody):
            name = self.model.body(b).name
            for side in SIDES:
                if name.startswith(f"openarm_{side}_"):
                    out[b] = (side, any(t in name for t in FINGER_BODY_TAGS))
        return out

    def _is_free_object(self, body: int) -> bool:
        m = self.model
        start, num = m.body_jntadr[body], m.body_jntnum[body]
        return any(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(start, start + num))

    def _contact_kind(self, b1: int, b2: int) -> str:
        r1, r2 = self._robot_bodies.get(b1), self._robot_bodies.get(b2)
        m = self.model
        if r1 and r2:
            if r1[0] != r2[0]:
                return "arm_arm"
            if (r1[1] and r2[1]) or m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1:
                return "ok"  # same gripper, or adjacent links
            return "self"
        if not r1 and not r2:
            return "ok"  # object on table etc.
        robot, other = (r1, b2) if r1 else (r2, b1)
        if self._is_free_object(other):
            return "ok"  # touching the object we manipulate is the point
        return "finger_env" if robot[1] else "env"

    # -------------------------------------------------------------- rendering
    def render(self, width: int = 480, height: int = 360, camera: str | int = -1) -> np.ndarray | None:
        """RGB image, or None when no offscreen GL context is available."""
        return self._render_thread.submit(self._render, width, height, camera).result()

    def _render(self, width: int, height: int, camera: str | int) -> np.ndarray | None:
        try:
            if self._renderer is None or (self._renderer.width, self._renderer.height) != (width, height):
                self._renderer = mujoco.Renderer(self.model, height, width)
            if self._render_data is None:
                self._render_data = mujoco.MjData(self.model)
            # Hold the lock only for the state copy: rendering can take 200 ms on a CPU-only GL stack.
            with self.lock:
                mujoco.mj_copyData(self._render_data, self.model, self.data)
            self._renderer.update_scene(self._render_data, camera=camera)
            return self._renderer.render().copy()
        except Exception:
            return None

    # ------------------------------------------------------------ conversions
    def _canonical_to_sim(self, q: np.ndarray) -> np.ndarray:
        out = q.astype(float).copy()
        for i in GRIPPER_IDX:
            g = float(np.clip(q[i], 0.0, 1.0))
            out[i] = self._grip_lo[i] + g * (self._grip_hi[i] - self._grip_lo[i])
        return out

    def _qpos_to_canonical(self, qpos: np.ndarray) -> np.ndarray:
        out = np.array([qpos[a] for a in self._qadr], dtype=float)
        for i in GRIPPER_IDX:
            span = self._grip_hi[i] - self._grip_lo[i]
            out[i] = float(np.clip((out[i] - self._grip_lo[i]) / span, 0.0, 1.0)) if span else 0.0
        return out
