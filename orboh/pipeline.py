"""Orchestrates the stages the website drives:

  preflight -> cameras -> demo -> DreamLab plan -> sim safety check -> approve & execute

Hard gates before any real motion:
  * the exact command sequence (approach + plan) passed static + MuJoCo checks
  * the robot has not moved since that check (else re-check)
  * an operator explicitly approved it
E-STOP is always accepted, even while another task runs.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from orboh.cameras.discovery import DiscoveryResult, assign_roles, changed_fraction, load_cameras_yaml, save_cameras_yaml
from orboh.cameras.sources import (SIM_DEVICES, CameraManager, CameraSource, RealSenseSource, SimCameraSource,
                                   list_realsense, to_jpeg)
from orboh.config import Config
from orboh.demos.dataset import LeRobotDemoDataset
from orboh.demos.sample import write_sample_dataset
from orboh.dreamlab.client import DreamLabClient, PlanRequest, make_dreamlab
from orboh.execution.executor import Executor, build_execution_trajectory
from orboh.robot import make_robot
from orboh.robot.mapping import JointMapping
from orboh.safety import CheckResult, SafetyReport, check_trajectory, fingerprint
from orboh.sim.mujoco_sim import MujocoSim
from orboh.types import ARM_IDX, JOINT_NAMES, Plan, Trajectory, joint_index

STEPS = ("preflight", "cameras", "demo", "plan", "check", "execute")


class Busy(RuntimeError):
    pass


class GateError(RuntimeError):
    pass


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.robot = make_robot(cfg)
        self.check_sim = MujocoSim(cfg.sim.scene)  # separate twin used only for checks
        self.lo, self.hi = self.check_sim.limits()
        self.mapping = JointMapping(cfg.robot.lerobot)
        self.cameras = CameraManager()
        self.dreamlab: DreamLabClient | None = None
        self.dreamlab_error = ""
        try:
            self.dreamlab = make_dreamlab(cfg.dreamlab)
        except Exception as e:
            self.dreamlab_error = str(e)

        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._task: str | None = None
        self._listeners: list[Callable[[dict], None]] = []
        self._logs: deque = deque(maxlen=300)
        self._record_proc: subprocess.Popen | None = None

        self.steps = {s: {"status": "pending", "detail": ""} for s in STEPS}
        self.preflight_results: list[CheckResult] = []
        self.discovery: DiscoveryResult | None = None
        self.demo = None
        self.plan: Plan | None = None
        self.exec_traj: Trajectory | None = None
        self.approach_steps = 0
        self.report: SafetyReport | None = None
        self.execution: dict | None = None
        self.progress = {"done": 0, "total": 0}
        self.run_dir: Path | None = None

    # ------------------------------------------------------------- plumbing
    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._listeners.append(fn)

    def _emit(self, kind: str, **data) -> None:
        event = {"type": kind, "t": time.time(), **data}
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:
                pass

    def log(self, msg: str, level: str = "info") -> None:
        entry = {"t": time.time(), "level": level, "msg": msg}
        self._logs.append(entry)
        self._emit("log", **entry)

    def _set_step(self, step: str, status: str, detail: str = "") -> None:
        self.steps[step] = {"status": status, "detail": detail}
        self._emit("state")

    def run_task(self, name: str, fn: Callable[[], None]) -> None:
        with self._lock:
            if self._task:
                raise Busy(f"busy with '{self._task}'")
            self._task = name

        def runner():
            self.log(f"▶ {name}")
            try:
                fn()
            except Exception as e:
                self.log(f"✖ {name} failed: {e}", "error")
                self.log(traceback.format_exc(limit=3), "debug")
            finally:
                with self._lock:
                    self._task = None
                self._emit("state")

        threading.Thread(target=runner, name=f"task-{name}", daemon=True).start()
        self._emit("state")

    def start(self) -> None:
        self.robot.connect()
        self.log(f"robot backend '{self.robot.name}' connected")
        saved = load_cameras_yaml(self.cfg.path(self.cfg.cameras.output))
        if saved:
            try:
                self._open_cameras(saved)
                self._set_step("cameras", "ok", f"loaded {self.cfg.cameras.output}")
            except Exception as e:
                self.log(f"could not open saved cameras: {e}", "warn")

    def shutdown(self) -> None:
        self.stop_event.set()
        self._stop_recording()
        self.cameras.close()
        self.robot.disconnect()

    # ------------------------------------------------------------ preflight
    def preflight(self) -> None:
        self._set_step("preflight", "running")
        res: list[CheckResult] = []
        try:
            q = self.robot.read_joints()
            res.append(CheckResult("robot_connection", True, f"{self.robot.name}: read {len(q)} joints"))
            m = self.cfg.safety.joint_limit_margin_rad
            out = [n for i, n in enumerate(JOINT_NAMES) if i in ARM_IDX and not (self.lo[i] + m <= q[i] <= self.hi[i] - m)]
            res.append(CheckResult("joints_within_limits", not out, "ok" if not out else f"near/over limit: {out}"))
        except Exception as e:
            res.append(CheckResult("robot_connection", False, str(e)))
        for role in self.cfg.cameras.expected_roles:
            jpg = self.cameras.snapshot(role, max_age_s=0) if role in self.cameras.sources else None
            res.append(CheckResult(f"camera_{role}", jpg is not None,
                                   "frame received" if jpg else "not assigned/no frame — run camera discovery"))
        if self.dreamlab:
            ok, msg = self.dreamlab.health()
            res.append(CheckResult("dreamlab_api", ok, f"{self.dreamlab.name}: {msg}"))
        else:
            res.append(CheckResult("dreamlab_api", False, self.dreamlab_error))
        res.append(CheckResult("sim_model", True, Path(self.check_sim.scene).name))
        eps = self.list_demos()
        res.append(CheckResult("demo_dataset", bool(eps), f"{len(eps)} episodes in {self.cfg.demos.dataset_root}",
                               warning=not eps))
        self.preflight_results = res
        hard_fail = [r for r in res if not r.passed and not r.warning]
        self._set_step("preflight", "fail" if hard_fail else "ok",
                       "; ".join(f"{r.name}: {r.detail}" for r in hard_fail) or "all checks passed")

    # -------------------------------------------------------------- cameras
    def _devices(self) -> list:
        return SIM_DEVICES if self.cfg.robot.backend == "sim" else list_realsense()

    def _make_source(self, device) -> CameraSource:
        if self.cfg.robot.backend == "sim":
            return SimCameraSource(self.robot.sim, device)
        c = self.cfg.cameras
        return RealSenseSource(device, c.width, c.height, c.fps)

    def _open_cameras(self, assignments: dict[str, str]) -> None:
        by_serial = {d.serial: d for d in self._devices()}
        sources = {}
        for role, serial in assignments.items():
            if serial not in by_serial:
                raise RuntimeError(f"camera {serial} for role {role} not connected")
            sources[role] = self._make_source(by_serial[serial])
        self.cameras.open(sources)

    def discover_cameras(self, wiggle: bool) -> None:
        self._set_step("cameras", "running")
        self.cameras.close()  # discovery needs every device to itself
        devices = self._devices()
        self.log(f"found {len(devices)} camera(s): " + ", ".join(f"{d.name} [{d.serial}]" for d in devices))
        scores = None
        if wiggle and len(devices) >= 2:
            scores = self._wiggle_test(devices)
        result = assign_roles(devices, self.cfg.cameras, scores)
        self.discovery = result
        for n in result.notes:
            self.log(n, "warn")
        self.set_camera_roles(result.assignments, result.method, result.confident)

    def _wiggle_test(self, devices) -> dict[str, float]:
        sources = {d.serial: self._make_source(d) for d in devices}
        try:
            grab = lambda: {s: src.read() for s, src in sources.items()}  # noqa: E731
            q0 = self.robot.read_joints()
            before = grab()
            noise = {s: changed_fraction(before[s], img) for s, img in grab().items()}
            j = joint_index(self.cfg.cameras.discovery.motion_joint)
            q1 = q0.copy()
            amp = self.cfg.cameras.discovery.motion_amplitude
            q1[j] = q1[j] - amp if q1[j] + amp > self.hi[j] - 0.05 else q1[j] + amp
            self.log(f"wiggle test: moving {self.cfg.cameras.discovery.motion_joint} by {amp}")
            self._safe_move_to(q1)
            time.sleep(0.3)
            after = grab()
            self._safe_move_to(q0)
            scores = {s: max(0.0, changed_fraction(before[s], after[s]) - noise[s]) for s in sources}
            self.log("wiggle scores (fraction of view changed): " +
                     ", ".join(f"{s}={v:.2f}" for s, v in scores.items()))
            return scores
        finally:
            for src in sources.values():
                src.close()

    def set_camera_roles(self, assignments: dict[str, str], method: str = "manual", confident: bool = True) -> None:
        devices = self._devices()
        result = DiscoveryResult(devices, assignments, method, confident,
                                 self.discovery.notes if self.discovery and method != "manual" else [],
                                 self.discovery.motion_scores if self.discovery and method != "manual" else {})
        self.discovery = result
        doc = save_cameras_yaml(result, self.cfg.cameras, self.cfg.path(self.cfg.cameras.output))
        self._open_cameras(assignments)
        self.log(f"camera roles ({method}): " + ", ".join(f"{r}={s}" for r, s in assignments.items()))
        self.log(f"for lerobot-record: --robot.cameras='{doc['lerobot_record_cameras_arg']}'", "debug")
        self._set_step("cameras", "ok" if confident else "warn",
                       f"{method}" + ("" if confident else " — please confirm roles from the thumbnails"))

    # ---------------------------------------------------------------- demos
    def _dataset(self) -> LeRobotDemoDataset:
        return LeRobotDemoDataset(self.cfg.path(self.cfg.demos.dataset_root), self.mapping)

    def list_demos(self) -> list[dict]:
        try:
            return self._dataset().list_episodes()
        except FileNotFoundError:
            return []

    def make_sample_demos(self) -> None:
        root = self.cfg.path(self.cfg.demos.dataset_root)
        if (root / "meta" / "info.json").exists():
            raise GateError(f"{root} already has a dataset; not overwriting")
        write_sample_dataset(root, self.mapping, self.check_sim.home())
        self.log(f"wrote synthetic demo dataset to {root}")

    def record_demo(self, mode: str) -> None:
        if mode not in ("dry", "run"):
            raise ValueError("mode must be dry or run")
        cmd = self.cfg.recording.command.format(mode=mode)
        self.cameras.close()  # lerobot-record must own the RealSense devices
        self.log(f"recording via: {cmd}")
        self._set_step("demo", "running", f"teleop recording ({mode})")
        try:
            self._record_proc = subprocess.Popen(
                shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in self._record_proc.stdout:
                self.log(line.rstrip(), "rec")
            code = self._record_proc.wait()
            self._set_step("demo", "ok" if code == 0 else "fail", f"recording exited with code {code}")
        finally:
            self._record_proc = None
            saved = load_cameras_yaml(self.cfg.path(self.cfg.cameras.output))
            if saved:
                self._open_cameras(saved)

    def _stop_recording(self) -> None:
        if self._record_proc and self._record_proc.poll() is None:
            self._record_proc.terminate()
            self.log("recording process terminated", "warn")

    # ----------------------------------------------------------------- plan
    def make_plan(self, episode: int, task: str) -> None:
        if not self.dreamlab:
            raise GateError(f"DreamLab not configured: {self.dreamlab_error}")
        self._set_step("plan", "running")
        self.plan = self.report = self.exec_traj = self.execution = None
        self._set_step("check", "pending")
        self._set_step("execute", "pending")
        self.demo = self._dataset().load(episode)
        self._set_step("demo", "ok", f"episode {episode}: {self.demo.task} ({self.demo.trajectory.duration:.1f}s)")
        images = {r: jpg for r in self.cameras.sources if (jpg := self.cameras.snapshot(r, max_age_s=0))}
        req = PlanRequest(task=task or self.demo.task, demo=self.demo, joints=self.robot.read_joints(), images=images)
        self.log(f"calling DreamLab ({self.dreamlab.name}) with demo {episode}, {len(images)} image(s)")
        t0 = time.time()
        self.plan = self.dreamlab.plan(req)
        tr = self.plan.trajectory
        self.run_dir = self.cfg.path(self.cfg.runs_dir) / time.strftime(f"%Y%m%d-%H%M%S_ep{episode}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plan.json").write_text(json.dumps(
            {"task": req.task, "episode": episode, "source": self.plan.source, "meta": self.plan.meta,
             "trajectory": tr.to_dict()}))
        self._set_step("plan", "ok", f"{self.plan.source}: {len(tr)} steps, {tr.duration:.1f}s "
                                     f"(took {time.time() - t0:.1f}s)")

    # ---------------------------------------------------------------- check
    def check(self, speed_scale: float = 1.0) -> None:
        if not self.plan:
            raise GateError("no plan yet")
        speed_scale = float(np.clip(speed_scale, 0.1, 1.0))
        self._set_step("check", "running", "static checks + MuJoCo rollout")
        start_q = self.robot.read_joints()
        self.exec_traj, self.approach_steps = build_execution_trajectory(
            start_q, self.plan.trajectory, self.cfg.safety, speed_scale)
        self.report = check_trajectory(self.exec_traj, self.check_sim, self.cfg.safety, start_q)
        self.report.sim["speed_scale"] = speed_scale
        for c in self.report.checks:
            self.log(f"{'✔' if c.passed else '✖'} {c.name}: {c.detail}", "info" if c.passed else "error")
        if self.run_dir:
            (self.run_dir / "safety_report.json").write_text(json.dumps(self.report.to_dict(), indent=2))
            for i, img in enumerate(self.report.artifacts.get("frames", [])):
                (self.run_dir / f"sim_preview_{i}.jpg").write_bytes(to_jpeg(img))
        warn = any(c.warning for c in self.report.checks)
        self._set_step("check", "ok" if self.report.passed and not warn else ("warn" if self.report.passed else "fail"),
                       "safe to execute — awaiting approval" if self.report.passed else "blocked: see failed checks")

    def sim_frame(self, i: int) -> bytes | None:
        frames = self.report.artifacts.get("frames", []) if self.report else []
        return to_jpeg(frames[i]) if 0 <= i < len(frames) else None

    # -------------------------------------------------------------- execute
    def gate(self) -> None:
        if not self.plan or not self.exec_traj:
            raise GateError("no checked plan")
        if self.cfg.safety.require_sim_pass and not (self.report and self.report.passed):
            raise GateError("sim safety check has not passed")
        if self.report.fingerprint != fingerprint(self.exec_traj):
            raise GateError("trajectory changed after the check; re-run the check")
        drift = float(np.max(np.abs(self.robot.read_joints() - np.asarray(self.report.start_q))[list(ARM_IDX)]))
        if drift > self.cfg.safety.max_state_drift_rad:
            raise GateError(f"robot moved {drift:.3f} rad since the check; re-run the check")

    def execute(self, approved: bool) -> None:
        if self.cfg.safety.require_manual_approval and not approved:
            raise GateError("execution needs explicit operator approval")
        self.gate()
        self.stop_event.clear()
        self._set_step("execute", "running", f"{len(self.exec_traj)} steps on {self.robot.name}")
        result = self._executor().run(self.exec_traj, self.stop_event)
        self.execution = result.__dict__
        if self.run_dir:
            (self.run_dir / "execution.json").write_text(json.dumps(self.execution, indent=2))
        self.log(("✔ " if result.completed else "✖ ") + result.reason, "info" if result.completed else "error")
        self._set_step("execute", "ok" if result.completed else "fail", result.reason)

    def _executor(self) -> Executor:
        def progress(done, total):
            self.progress = {"done": done, "total": total}
            self._emit("progress", **self.progress)
        return Executor(self.robot, self.cfg.safety, self.lo, self.hi, progress)

    def _safe_move_to(self, q_target: np.ndarray) -> None:
        """Small utility moves (home, wiggle) still go through static + sim checks."""
        q0 = self.robot.read_joints()
        traj, _ = build_execution_trajectory(q0, Trajectory(np.vstack([q_target, q_target]), self.cfg.robot.control_hz),
                                             self.cfg.safety)
        report = check_trajectory(traj, self.check_sim, self.cfg.safety, q0)
        if not report.passed:
            bad = "; ".join(f"{c.name}: {c.detail}" for c in report.checks if not c.passed)
            raise GateError(f"move refused by safety check: {bad}")
        self.stop_event.clear()
        result = self._executor().run(traj, self.stop_event)
        if not result.completed:
            raise GateError(f"move aborted: {result.reason}")

    def go_home(self) -> None:
        self._safe_move_to(self.check_sim.home())
        self.log("at home pose")

    def estop(self) -> None:
        self.stop_event.set()
        self._stop_recording()
        self.log("E-STOP pressed — motion halted, holding position", "error")
        self._emit("state")

    # ---------------------------------------------------------------- state
    def state(self) -> dict:
        try:
            joints = [round(float(v), 4) for v in self.robot.read_joints()]
        except Exception:
            joints = []
        return {
            "task": self._task,
            "estopped": self.stop_event.is_set(),
            "robot_backend": self.robot.name,
            "dreamlab_backend": self.dreamlab.name if self.dreamlab else None,
            "steps": self.steps,
            "joints": joints,
            "preflight": [c.__dict__ for c in self.preflight_results],
            "cameras": {
                "roles": list(self.cameras.sources),
                "discovery": self.discovery.to_dict() if self.discovery else None,
            },
            "demo": {"index": self.demo.index, "task": self.demo.task} if self.demo else None,
            "plan": {"source": self.plan.source, "steps": len(self.plan.trajectory),
                     "duration_s": round(self.plan.trajectory.duration, 2)} if self.plan else None,
            "report": self.report.to_dict() | {"frames": len(self.report.artifacts.get("frames", []))}
            if self.report else None,
            "approach_steps": self.approach_steps,
            "execution": self.execution,
            "progress": self.progress,
            "recording": self._record_proc is not None,
            "safety": {"require_manual_approval": self.cfg.safety.require_manual_approval},
            "run_dir": str(self.run_dir) if self.run_dir else None,
        }

    def logs(self) -> list[dict]:
        return list(self._logs)

    def trajectory_data(self, max_points: int = 400) -> dict | None:
        if not self.exec_traj and not self.plan:
            return None
        traj = self.exec_traj or self.plan.trajectory
        step = max(1, len(traj) // max_points)
        out = {"hz": traj.hz / step, "names": list(JOINT_NAMES), "command": traj.positions[::step].round(4).tolist(),
               "approach_steps": self.approach_steps // step if self.exec_traj else 0}
        measured = self.report.artifacts.get("measured") if self.report else None
        if measured is not None:
            out["sim"] = measured[::step].round(4).tolist()
        return out
