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


class CamerasConfig(BaseModel):
    expected_roles: list[str] = Field(default_factory=lambda: ["center", "right_wrist"])
    width: int = 640
    height: int = 480
    fps: int = 30
    pinned_serials: dict[str, str] = Field(default_factory=dict)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    output: str = "config/cameras.yaml"


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
    max_check_failures: int = 3  # consecutive rejected chunks before the run halts
    starve_s: float = 3.0  # no usable chunk for this long -> halt


class DemosConfig(BaseModel):
    dataset_root: str = "data/demos"


class RecordingConfig(BaseModel):
    command: str = ""


class SimConfig(BaseModel):
    engine: Literal["mujoco"] = "mujoco"
    scene: str = "pedestal/move_puck_scene.xml"


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


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    token: str = ""  # Filled from .env


class Config(BaseModel):
    robot: RobotConfig = Field(default_factory=RobotConfig)
    cameras: CamerasConfig = Field(default_factory=CamerasConfig)
    dreamlab: DreamLabConfig = Field(default_factory=DreamLabConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
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
