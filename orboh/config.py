"""Pipeline configuration: config/pipeline.yaml for settings, .env for secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"


class LeRobotConfig(BaseModel):
    robot_type: str = "bi_openarm_follower"
    robot_kwargs: dict = Field(default_factory=dict)
    joint_key: str = "{side}_joint_{index}.pos"
    gripper_key: str = "{side}_gripper.pos"
    joint_units: Literal["radians", "degrees"] = "radians"
    joint_signs: dict[str, float] = Field(default_factory=dict)
    joint_offsets: dict[str, float] = Field(default_factory=dict)
    gripper_closed: float = 0.0
    gripper_open: float = 1.0


class RobotBridgeConfig(BaseModel):
    """Where scripts/robot_bridge.py listens (backend: bridge)."""

    endpoint: str = "tcp://127.0.0.1:5560"
    timeout_ms: int = 2000


class RobotConfig(BaseModel):
    backend: Literal["sim", "lerobot", "bridge"] = "sim"
    control_hz: float = 30
    lerobot: LeRobotConfig = Field(default_factory=LeRobotConfig)
    bridge: RobotBridgeConfig = Field(default_factory=RobotBridgeConfig)


class DiscoveryConfig(BaseModel):
    motion_joint: str = "right_gripper"
    motion_amplitude: float = 0.6


class PerceptionConfig(BaseModel):
    """Classical block finder on the depth camera (orboh/perception/block_finder.py)."""

    roi: list[int] = Field(default_factory=lambda: [0, 290, 640, 480])  # x0, y0, x1, y1 in colour pixels: the carpet only
    min_height_mm: float = 15  # above the fitted table plane
    max_height_mm: float = 130
    plane_tol_mm: float = 6  # RANSAC inlier distance
    min_pixels: int = 250
    footprint_range_mm: list[float] = Field(default_factory=lambda: [25, 130])  # sane object sizes; outside = rejected
    height_range_mm: list[float] = Field(default_factory=lambda: [20, 110])
    match_tol_mm: float = 14
    # Known pieces (Kitakyushu wood-L1 scene): footprint sides (mm, sorted) and height; lying variants listed separately.
    pieces: dict[str, list[float]] = Field(default_factory=lambda: {
        "cube": [47, 47, 47], "cylinder_standing": [46, 46, 98], "cylinder_lying": [46, 98, 46]})


class GraspConfig(BaseModel):
    """Pick-and-place planner (orboh/planning/grasp.py). Numbers marked (demos) come from the 8 recorded wood-L1 episodes (work/18)."""

    tcp_forward_mm: float = 37  # fingertip contact point beyond the finger-mesh midpoint, along the gripper axis (calibration fit)
    approach_pitch_deg: float = 22  # downward tilt of the approach (demos: 10-31, median ~22)
    carry_pitch_deg: float = 10  # flatter from the lift on: the 22 deg tilt runs into joint limits (j2 is nearly at its limit in the demos too)
    joint_margin_rad: float = 0.07  # IK stays this far inside the joint limits (the safety check demands 0.05)
    pregrasp_back_mm: float = 90  # stop this far back along the approach before closing in
    lift_mm: float = 110  # straight lift after the grasp (higher is out of reach this close to the body at this tilt)
    carry_z: float = 0.58  # tool-point height over the box; the diagonal from the lift clears the rim (0.455) by ~3 cm
    gripper_open: float = 0.74  # canonical 0 closed .. 1 open; -48 deg of -65 (demos opened to -42..-54)
    gripper_close: float = 0.25  # -16 deg: past the contact angle (-32 deg = 0.49), so the controller squeezes
    arm_split_y: float = -0.04  # block y above this -> left arm (demos: left picked y >= -0.03, right y <= -0.08)
    place_x: float = 0.495  # release point over the box (demos: x 0.49-0.51)
    place_y: float = 0.07  # +y for the left arm, -y for the right (demos: left +0.07, right -0.05..-0.12)
    place_z: float = 0.50  # tool point height at release (demos 0.47-0.53; box rim 0.455)
    # Natural arm posture at the grasp (rad, joints 1-7): median of the demos' follower joints (left from 5 picks, right from 3).
    # The IK is gently pulled toward it so the elbow stays down and out like the operator's, not folded into the torso.
    posture_left: list[float] = Field(default_factory=lambda: [0.01, 0.10, 0.24, 1.25, 0.08, -0.08, 0.05])
    posture_right: list[float] = Field(default_factory=lambda: [0.13, 0.00, -0.05, 1.15, -0.09, 0.08, 0.07])
    posture_cost: float = 0.02
    clearance_mm: float = 8  # IK keeps the arm at least this far from the torso, box and table (fingers excepted on the table)
    grasp_lift_mm: float = 6  # close on the block this far above its centre: the hand base touches the table in the demos otherwise
    ik_pos_tol_mm: float = 3
    ik_ori_tol_deg: float = 8
    ik_max_iters: int = 300
    ready_speed_rad_s: float = 0.4  # peak joint speed of the unfold from the start pose to the natural posture
    tool_speed_m_s: float = 0.12  # peak tool-point speed along a straight segment (min-jerk)
    tool_rot_speed_rad_s: float = 0.6
    gripper_time_s: float = 1.0  # pause while the gripper opens or closes


class CamerasConfig(BaseModel):
    expected_roles: list[str] = Field(default_factory=lambda: ["center", "right_wrist"])
    depth_roles: list[str] = Field(default_factory=list)  # roles that also stream depth (needed by `pick`)
    width: int = 640
    height: int = 480
    fps: int = 30
    pinned_serials: dict[str, str] = Field(default_factory=dict)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    output: str = "config/cameras.yaml"
    extrinsics: str = "config/camera_extrinsics.yaml"  # camera -> robot base, per machine (scripts/calibrate_camera.py)


class DreamLabConfig(BaseModel):
    backend: Literal["mock", "http"] = "mock"
    timeout_s: float = 120
    plan_endpoint: str = "/plan"
    health_endpoint: str = "/health"
    send_images: bool = True
    # Filled from .env
    api_key: str = ""
    base_url: str = ""


class PolicyConfig(BaseModel):
    """Closed-loop Dream-1 policy: camera frames + joints in, action chunks out (see work/14-policy.md)."""

    backend: Literal["mock", "dream"] = "mock"
    checkpoint: str = "dreamlabs/dream-1-beta-flash"
    prompt: str = "pick up the object"
    ik_arm: str = "openarm2_faas"  # dream client IK arm spec; the base model answers in tool poses
    action_hz: float = 20
    execution_horizon: int = 30
    timeout_ms: int = 15000
    strict: bool = True
    max_duration_s: float = 30
    obs_hz: float = 5  # frames + joints handed to the client; it keeps 5 per second anyway
    image_width: int = 320
    image_height: int = 176
    jpeg_quality: int = 85
    # Dream camera slot -> Orboh camera role (real robot). A slot with no camera gets a black frame.
    camera_map: dict[str, str | None] = Field(
        default_factory=lambda: {"overhead": "center", "wrist_l": "left_wrist", "wrist_r": "right_wrist"})
    # Dream camera slot -> MuJoCo camera name (sim backend)
    sim_cameras: dict[str, str] = Field(
        default_factory=lambda: {"overhead": "tablecam", "wrist_l": "camera_wrist_left",
                                 "wrist_r": "camera_wrist_right"})
    # Gripper: "model" = use what Dream-1 commands; "demos" = snap its open/close decision to the values measured in the 8 recorded
    # demos (work/18): open ~ -48 deg, squeeze ~ -16 deg (the follower stalls on the block at ~ -32 deg). Dream is trained on other
    # grippers, so its raw numbers are untested on OpenArm.
    speed_scale: float = 0.3  # how fast the arms may move, as a fraction of the safety limits (0.1 slowest .. 1.0 full speed)
    gripper_mode: Literal["model", "demos"] = "model"
    demo_gripper_open: float = 0.74  # canonical 0 closed .. 1 open
    demo_gripper_close: float = 0.25
    gripper_close_below: float = 0.40  # hysteresis on Dream's value: below this it means close ...
    gripper_open_above: float = 0.60  # ... above this it means open; in between keeps the last decision
    require_dry_run_first: bool = True  # a real run needs a dry-run with the same prompt whose first chunk passed
    max_check_failures: int = 8  # consecutive rejected plans before the run gives up: each one is a retry, Dream-1 is asked again
    retry_hint: bool = True  # on a rejection tell Dream-1 what went wrong (added to its prompt) until it produces a safe plan
    starve_s: float = 25.0  # no usable chunk for this long -> halt (a Dream round trip is ~2.4 s, so this allows the retries)


class DemosConfig(BaseModel):
    dataset_root: str = "data/demos"


class RecordingConfig(BaseModel):
    command: str = ""


class SimConfig(BaseModel):
    engine: Literal["mujoco"] = "mujoco"
    scene: str = "pedestal/move_puck_scene.xml"
    torso_tolerance_m: float = 0.0  # arm-vs-torso overlap allowed (model is stricter than reality); 0 = strict
    # Open-topped box on the table: center [x, y], size [depth_x, width_y, height] (m). None = no box in the twin.
    box: dict | None = None
    table_drop_m: float = 0.0  # lower the twin's table by this much (twin vs real table height; see work/07)


class SafetyConfig(BaseModel):
    joint_limit_margin_rad: float = 0.05
    max_joint_velocity_rad_s: float = 1.2
    max_joint_accel_rad_s2: float = 8.0
    max_gripper_velocity: float = 3.0
    approach_speed_rad_s: float = 0.3
    max_plan_duration_s: float = 120
    sim_max_tracking_error_rad: float = 0.25
    runtime_max_tracking_error_rad: float = 0.35
    max_state_drift_rad: float = 0.05
    watchdog_s: float = 0.25
    allow_finger_env_contact: bool = True
    require_sim_pass: bool = True
    require_manual_approval: bool = True
    conform_plan: bool = True  # slew-limit + clip every plan to the limits before the checks (a leader demo is jerky)


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    token: str = ""  # Filled from .env


class Config(BaseModel):
    robot: RobotConfig = Field(default_factory=RobotConfig)
    cameras: CamerasConfig = Field(default_factory=CamerasConfig)
    dreamlab: DreamLabConfig = Field(default_factory=DreamLabConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    grasp: GraspConfig = Field(default_factory=GraspConfig)
    demos: DemosConfig = Field(default_factory=DemosConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    runs_dir: str = "runs"

    def path(self, p: str) -> Path:
        """Resolve a config path relative to the project root."""
        path = Path(p).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path | None = None) -> Config:
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(path or os.environ.get("ORBOH_CONFIG") or DEFAULT_CONFIG)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = Config.model_validate(data or {})
    cfg.dreamlab.api_key = os.environ.get("DREAMLAB_API_KEY", "")
    cfg.dreamlab.base_url = os.environ.get("DREAMLAB_BASE_URL", "").rstrip("/")
    cfg.web.token = os.environ.get("ORBOH_WEB_TOKEN", "")
    return cfg
