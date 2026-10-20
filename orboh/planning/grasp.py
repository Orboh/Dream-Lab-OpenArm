"""Block position -> pick-and-place joint trajectory (simulator model only; nothing here touches the robot).

For a block at `xyz` (robot base frame) the planner:
  1. picks the arm nearer the block (rule taken from the demos),
  2. builds tool-point waypoints: pre-grasp (backed off along the approach) -> grasp -> close -> lift -> over the box
     -> release -> retreat, with the gripper approaching forward and ~22 deg downward and its fingers across a block face,
  3. solves each waypoint with differential IK (`mink`) on the MuJoCo twin, seeded from the previous one,
  4. moves the tool point along straight lines between waypoints (minimum-jerk pacing), solving the IK at every step, so the
     arm never takes a joint-space shortcut through the torso or the box.
The result is an ordinary `Trajectory`: it still goes through `conform_plan`, the static checks, the MuJoCo rollout and the
approval gate. The tool point is the fingertip contact point: the midpoint of the two finger meshes plus `tcp_forward_mm` along
the gripper axis (that offset was fitted from the demos, work/17).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import orboh.sim  # noqa: F401  (must come before mujoco is imported: sets MUJOCO_GL, see orboh/sim/__init__.py)

import mink  # noqa: E402
import mujoco  # noqa: E402

from orboh.config import GraspConfig
from orboh.sim.mujoco_sim import MujocoSim
from orboh.types import GRIPPER_IDX, JOINT_NAMES, Trajectory


class GraspPlanError(RuntimeError):
    pass


@dataclass
class GraspPlan:
    trajectory: Trajectory
    arm: str
    waypoints: list[dict] = field(default_factory=list)  # name, tool point (m), IK position error (mm), orientation error (deg)

    def summary(self) -> dict:
        return {"arm": self.arm, "duration_s": round(self.trajectory.duration, 1), "waypoints": self.waypoints}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _triad(a: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Right-handed frame (columns): a, s made orthogonal to a, a x s."""
    s = _unit(s - (s @ a) * a)
    return np.stack([a, s, np.cross(a, s)], axis=1)


class GraspPlanner:
    def __init__(self, sim: MujocoSim, cfg: GraspConfig, hz: float):
        self.sim, self.cfg, self.hz = sim.planning_copy(), cfg, hz
        self.model = self.sim.model

    # ------------------------------------------------------------------ gripper geometry
    def _tool_frame(self, side: str) -> dict:
        """The tool point and gripper axes in the wrist body's local frame, from forward kinematics at the current sim pose."""
        m, d = self.model, self.sim.data
        body = m.body(f"openarm_{side}_ee_base_link").id
        fing = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(f"openarm_{side}_ee_")
                and "finger" in m.body(m.geom_bodyid[g]).name and m.geom_contype[g]]
        mid = np.mean([d.geom_xpos[g] for g in fing], axis=0)
        wrist, R = d.xpos[body].copy(), d.xmat[body].reshape(3, 3).copy()
        a = _unit(mid - wrist)
        sep = d.geom_xpos[fing[0]] - d.geom_xpos[fing[-1]]  # same definition as the demo analysis (work/18)
        a_l, sep_l = R.T @ a, R.T @ sep
        tool_local = R.T @ (mid + self.cfg.tcp_forward_mm / 1000 * a - wrist)
        return {"body": f"openarm_{side}_ee_base_link", "tool_local": tool_local, "triad_local": _triad(a_l, sep_l)}

    def _target_rotation(self, side: str, heading: float, tool: dict, pitch_deg: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(wrist rotation, approach axis in the world) for approaching along `heading` with a downward tilt."""
        p = np.radians(self.cfg.approach_pitch_deg if pitch_deg is None else pitch_deg)
        a_w = np.array([np.cos(p) * np.cos(heading), np.cos(p) * np.sin(heading), -np.sin(p)])
        sign = -1.0 if side == "left" else 1.0  # demos: separation axis ~ -y for the left arm, +y for the right
        s_w = sign * np.array([-np.sin(heading), np.cos(heading), 0.0])
        return _triad(a_w, s_w) @ tool["triad_local"].T, a_w

    def _collision_limit(self, side: str) -> mink.CollisionAvoidanceLimit | None:
        """Keep the planned arm out of the torso, the box and the table (fingers may touch the table: that is the grasp)."""
        m = self.model
        name = lambda g: m.geom(g).name or ""
        mount = ("_base_link", "_link1", "_link2")
        arm = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith(f"openarm_{side}_")
               and not m.body(m.geom_bodyid[g]).name.endswith(mount)]
        fingers = [g for g in arm if "finger" in m.body(m.geom_bodyid[g]).name]
        torso = [g for g in range(m.ngeom) if name(g).startswith("openarm_body_link0_collision")]
        box = [g for g in range(m.ngeom) if name(g).startswith("box_")]
        table = [m.geom("table_top").id]
        pairs = [(torso, arm), (box, arm), (table, [g for g in arm if g not in fingers])]
        pairs = [(a, b) for a, b in pairs if a and b]
        if not pairs:
            return None
        return mink.CollisionAvoidanceLimit(m, pairs, minimum_distance_from_collisions=self.cfg.clearance_mm / 1000,
                                            collision_detection_distance=0.08)

    # ------------------------------------------------------------------ IK
    def _ik(self, side: str, tool: dict, tool_xyz: np.ndarray, R: np.ndarray, qpos0: np.ndarray,
            loose: bool = False) -> tuple[np.ndarray, float, float]:
        """`loose`: a point along a path (needs continuity, not precision); the end of every segment is solved strictly."""
        c = self.cfg
        pos_tol, ori_tol = (4 * c.ik_pos_tol_mm, 3 * c.ik_ori_tol_deg) if loose else (c.ik_pos_tol_mm, c.ik_ori_tol_deg)
        wrist_xyz = tool_xyz - R @ tool["tool_local"]
        conf = mink.Configuration(self.model, q=qpos0.copy())
        task = mink.FrameTask(tool["body"], "body", position_cost=1.0, orientation_cost=0.4, lm_damping=1e-3)
        task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), wrist_xyz))
        target = qpos0.copy()
        for j, v in enumerate(c.posture_left if side == "left" else c.posture_right):
            target[self.sim._qadr[(0 if side == "left" else 8) + j]] = v
        posture = mink.PostureTask(self.model, cost=c.posture_cost)
        posture.set_target(target)
        limits = [mink.ConfigurationLimit(self.model, min_distance_from_limits=c.joint_margin_rad)]
        if (cl := self._collision_limit(side)) is not None:
            limits.append(cl)
        pe = oe = np.inf
        for _ in range(c.ik_max_iters):
            conf.integrate_inplace(mink.solve_ik(conf, [task, posture], 0.05, "daqp", limits=limits), 0.05)
            err = task.compute_error(conf)
            pe, oe = float(np.linalg.norm(err[:3])) * 1000, float(np.degrees(np.linalg.norm(err[3:])))
            if pe < pos_tol / 2 and oe < ori_tol / 2:
                break
        if pe > pos_tol or oe > ori_tol:
            raise GraspPlanError(f"IK did not reach {np.round(tool_xyz, 3)} for the {side} arm "
                                 f"(position error {pe:.1f} mm, orientation {oe:.1f} deg): out of reach or a limit is in the way")
        return conf.q.copy(), pe, oe

    # ------------------------------------------------------------------ planning
    def choose_arm(self, block_xyz: np.ndarray) -> str:
        return "left" if block_xyz[1] > self.cfg.arm_split_y else "right"

    def _tool_pose(self, side: str, tool: dict) -> tuple[np.ndarray, np.ndarray]:
        """Current tool point (m) and wrist rotation in the world, from the sim's current state."""
        d = self.sim.data
        body = self.model.body(tool["body"]).id
        R = d.xmat[body].reshape(3, 3).copy()
        return d.xpos[body] + R @ tool["tool_local"], R

    def _fk_tool(self, side: str, tool: dict, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.sim.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.sim.data)
        return self._tool_pose(side, tool)

    def _ready_posture(self, side: str, qpos: np.ndarray) -> np.ndarray:
        q = qpos.copy()
        for j, v in enumerate(self.cfg.posture_left if side == "left" else self.cfg.posture_right):
            q[self.sim._qadr[(0 if side == "left" else 8) + j]] = v
        return q

    def _move(self, side: str, tool: dict, qpos: np.ndarray, p0, R0, p1, R1) -> list[np.ndarray]:
        """IK along the straight tool-point line p0 -> p1 (rotation interpolated R0 -> R1): one qpos per ~0.1 s."""
        c = self.cfg
        dist = float(np.linalg.norm(p1 - p0))
        rot = float(np.linalg.norm((mink.SO3.from_matrix(R0).inverse() @ mink.SO3.from_matrix(R1)).log()))
        dur = max(1.875 * dist / c.tool_speed_m_s, 1.875 * rot / c.tool_rot_speed_rad_s, 0.5)
        n = max(2, int(np.ceil(dur * self.IK_HZ)))
        so0, rel = mink.SO3.from_matrix(R0), mink.SO3.from_matrix(R0).inverse() @ mink.SO3.from_matrix(R1)
        out, q = [], qpos
        for k in range(1, n + 1):
            u = k / n
            s = 10 * u**3 - 15 * u**4 + 6 * u**5  # minimum-jerk progress
            R = (so0 @ mink.SO3.exp(s * rel.log())).as_matrix()
            q, _, _ = self._ik(side, tool, p0 + s * (p1 - p0), R, q, loose=k < n)
            out.append(q)
        return out

    IK_HZ = 10

    def plan(self, q0: np.ndarray, block_xyz: np.ndarray, block_yaw: float = 0.0, arm: str | None = None) -> GraspPlan:
        """`block_xyz`: centre of the block in the base frame (m); `block_yaw`: direction of one of its faces in the base xy plane (rad)."""
        c = self.cfg
        q0 = np.asarray(q0, float)
        block_xyz = np.asarray(block_xyz, float)
        side = arm or self.choose_arm(block_xyz)
        sign = 1.0 if side == "left" else -1.0
        self.sim.reset(q0)
        qpos = self.sim.data.qpos.copy()
        tool = self._tool_frame(side)
        p, R = self._tool_pose(side, tool)
        shoulder = self.sim.data.xpos[self.model.body(f"openarm_{side}_base_link").id]
        heading_nat = float(np.arctan2(block_xyz[1] - shoulder[1], block_xyz[0] - shoulder[0]))
        # approach face-on: the block face normal closest to the natural heading (never more than 45 deg off it)
        heading = block_yaw + (np.pi / 2) * np.round((heading_nat - block_yaw) / (np.pi / 2))
        R_t, a_w = self._target_rotation(side, heading, tool)
        heading_box = float(np.arctan2(sign * c.place_y - shoulder[1], c.place_x - shoulder[0]))
        R_c, _ = self._target_rotation(side, heading_box, tool, c.carry_pitch_deg)  # flatter, aimed at the box
        up = np.array([0.0, 0.0, 1.0])
        place = np.array([c.place_x, sign * c.place_y, c.place_z])
        carry_z = c.carry_z
        # name, tool point, gripper value after the step (None = unchanged)
        steps = [("pregrasp", block_xyz - c.pregrasp_back_mm / 1000 * a_w, R_t, c.gripper_open),
                 ("grasp", block_xyz + c.grasp_lift_mm / 1000 * up, R_t, None), ("close", None, R_t, c.gripper_close),
                 ("lift", block_xyz + (c.lift_mm + c.grasp_lift_mm) / 1000 * up, R_c, None),
                 ("over_box", np.array([place[0], place[1], carry_z]), R_c, None),
                 ("release_pose", place, R_c, None), ("release", None, R_c, c.gripper_open),
                 ("retreat", np.array([place[0], place[1], carry_z]), R_c, None)]
        grip = float(q0[GRIPPER_IDX[0 if side == "left" else 1]])
        other = [i for i, n in enumerate(JOINT_NAMES) if not n.startswith(side)]
        gi = GRIPPER_IDX[0 if side == "left" else 1]

        def row(qp: np.ndarray, g: float) -> np.ndarray:
            v = self.sim._qpos_to_canonical(qp)
            v[other], v[gi] = q0[other], g
            return v

        rows, info, cur_g = [q0], [], grip
        # first unfold the arm to the operator's natural forward posture (joint space), then reach along straight lines:
        # a straight tool-point line from the folded start pose runs through the torso
        ready = self._ready_posture(side, qpos)
        arm_adr = [self.sim._qadr[(0 if side == "left" else 8) + j] for j in range(7)]
        travel = float(np.abs(ready[arm_adr] - qpos[arm_adr]).max())
        n = max(2, int(np.ceil(1.875 * travel / c.ready_speed_rad_s * self.IK_HZ)))
        for k in range(1, n + 1):
            u = k / n
            s = 10 * u**3 - 15 * u**4 + 6 * u**5
            rows.append(row(qpos + s * (ready - qpos), cur_g))
        qpos = ready
        p, R = self._fk_tool(side, tool, qpos)
        for name, xyz, R_goal, g in steps:
            if xyz is not None:  # an arm move along a straight line, IK solved all the way
                path = self._move(side, tool, qpos, p, R, xyz, R_goal)
                rows += [row(qp, cur_g) for qp in path]
                qpos, p, R = path[-1], xyz, R_goal
                pe, oe = self._pose_error(side, tool, qpos, xyz, R_goal)
                info.append({"name": name, "tool_point": [round(float(v), 4) for v in xyz],
                             "ik_pos_err_mm": round(pe, 2), "ik_ori_err_deg": round(oe, 1)})
            if g is not None:  # a gripper move: hold the pose and ramp the gripper
                n = max(2, int(round(c.gripper_time_s * self.IK_HZ)))
                last = rows[-1]
                for k in range(1, n + 1):
                    v = last.copy()
                    v[gi] = cur_g + (g - cur_g) * k / n
                    rows.append(v)
                cur_g = g
                if xyz is None:
                    info.append({"name": name, "gripper": round(g, 2)})
        # rows are 0.1 s apart: resample to the control rate (linear in joint space between close poses)
        rows = np.asarray(rows)
        t_ik = np.arange(len(rows)) / self.IK_HZ
        t_out = np.arange(0.0, t_ik[-1] + 1e-9, 1.0 / self.hz)
        dense = np.stack([np.interp(t_out, t_ik, rows[:, j]) for j in range(rows.shape[1])], axis=1)
        return GraspPlan(Trajectory(dense, self.hz), side, info)

    def _pose_error(self, side, tool, qpos, xyz, R) -> tuple[float, float]:
        conf = mink.Configuration(self.model, q=qpos)
        wrist = xyz - R @ tool["tool_local"]
        task = mink.FrameTask(tool["body"], "body", position_cost=1.0, orientation_cost=0.4)
        task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), wrist))
        e = task.compute_error(conf)
        return float(np.linalg.norm(e[:3])) * 1000, float(np.degrees(np.linalg.norm(e[3:])))
