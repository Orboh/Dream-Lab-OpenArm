"""Closed-loop Dream-1 policy client.

Dream-1 is a policy, not a planner: every few hundred ms it takes the latest camera frames, the joint state
and a text prompt, and returns a chunk of joint targets (about 2.4 s at 20 Hz). This module hides the
`dreamlabs-client` library behind a small interface in canonical Orboh units (see orboh.types):

  open(prompt) -> observe(q, jpegs) ... -> get_chunk() -> Trajectory (H, 16) at action_hz

The base model is trained on 6-DoF YAM arms, so for OpenArm the client library runs its own IK on the model's
end-effector answer (`ik` config, arm `openarm2_faas`, whose joint limits match the MuJoCo twin exactly).
"""

from __future__ import annotations

import base64
import io
import time
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from orboh.config import PolicyConfig
from orboh.types import NUM_JOINTS, Trajectory


class PolicyError(RuntimeError):
    def __init__(self, msg: str, retryable: bool = False, retry_after_s: float | None = None):
        super().__init__(msg)
        self.retryable, self.retry_after_s = retryable, retry_after_s


class PolicyClient(ABC):
    name = "policy"
    action_hz: float = 20.0

    @abstractmethod
    def open(self, prompt: str) -> None:
        """Start an episode for `prompt`."""

    @abstractmethod
    def observe(self, q: np.ndarray, jpegs: dict[str, bytes | None]) -> None:
        """Hand over one tick: canonical joints (16,) and a JPEG per Dream camera slot (None = no camera)."""

    @abstractmethod
    def get_chunk(self) -> Trajectory:
        """Block for the next action chunk, canonical (H, 16) at `action_hz`."""

    def refill_due(self, executed: int, remaining: int) -> bool:
        """Whether the next get_chunk() should start now (called once per executed row)."""
        return remaining <= 0

    def health(self) -> tuple[bool, str]:
        return True, "ok"

    def timing(self) -> dict | None:
        return None

    def close(self) -> None:
        pass


def _canonical_state(q: np.ndarray) -> dict[str, np.ndarray]:
    """Canonical (16,) -> Dream's `state.*` keys, float32: arm joints in radians, `*_grip` 0 closed .. 1 open.

    With client-side IK the client derives the gripper drive and tool pose itself, so only these are sent.
    """
    q = np.asarray(q, dtype=np.float32)
    return {
        "state.left_arm": q[0:7].copy(), "state.left_grip": q[7:8].copy(),
        "state.right_arm": q[8:15].copy(), "state.right_grip": q[15:16].copy(),
    }


def chunk_to_canonical(action: dict, hz: float) -> Trajectory:
    """Dream's `q_l_chunk` / `q_r_chunk` ([H, 7 joints + gripper]) -> canonical (H, 16) trajectory."""
    try:
        left, right = np.asarray(action["q_l_chunk"], dtype=float), np.asarray(action["q_r_chunk"], dtype=float)
    except (KeyError, TypeError, ValueError) as e:
        raise PolicyError(f"reply has no usable q_l_chunk/q_r_chunk: {e}; keys={list(action)[:12]}") from e
    if left.ndim != 2 or left.shape != right.shape or left.shape[1] != 8 or len(left) == 0:
        raise PolicyError(f"unexpected chunk shapes: left {left.shape}, right {right.shape} (want [H, 8] each)")
    return Trajectory(np.concatenate([left, right], axis=1), hz)  # layout: l1..l7, l_grip, r1..r7, r_grip


class DreamPolicyClient(PolicyClient):
    """Real Dream API through the official `dreamlabs-client` library (Apache-2.0)."""

    name = "dream"

    def __init__(self, cfg: PolicyConfig, api_key: str):
        try:
            import dream
        except ImportError as e:
            raise PolicyError("the Dream client is not installed: bash scripts/setup_env.sh --dream "
                              "(Python 3.11-3.13)") from e
        if not api_key:
            raise PolicyError("DREAMLAB_API_KEY is not set in .env")
        self.cfg = cfg
        self.action_hz = cfg.action_hz
        self._dream = dream
        self._client = dream.DreamClient(api_key=api_key)
        self._black: bytes | None = None
        self._slots = list(cfg.camera_map)

    # Dream's embodiment spec: what each observation / action key must look like (checked locally).
    def _embodiment(self) -> dict:
        state = {"left_arm": [0, 7], "left_grip": [7, 8], "right_arm": [8, 15], "right_grip": [15, 16],
                 # added by the client's IK from the arms above; declared so strict mode accepts them
                 "left_eef_xyz": [16, 19], "left_eef_rot6": [19, 25],
                 "right_eef_xyz": [25, 28], "right_eef_rot6": [28, 34]}
        # the model answers joints plus tool poses; the client's IK turns the poses into the chunk we use
        action = {"left_arm": [0, 7], "left_grip": [7, 8], "right_arm": [8, 15], "right_grip": [15, 16],
                  "left_eef_xyz": [16, 19], "left_eef_rot6": [19, 25],
                  "right_eef_xyz": [25, 28], "right_eef_rot6": [28, 34]}
        return {"state": state, "action": action,
                "video": {slot: f"observation.images.{slot}" for slot in self._slots}}

    def _config(self) -> dict:
        c = self.cfg
        return {"checkpoint": c.checkpoint, "timeout_ms": c.timeout_ms, "strict": c.strict,
                "action_hz": c.action_hz, "execution_horizon": c.execution_horizon,
                "ik": {"arm": c.ik_arm, "head": "eef"}, "embodiment": self._embodiment()}

    def open(self, prompt: str) -> None:
        self._prompt = prompt
        try:
            self._client.policy(self._config())
        except self._dream.DreamError as e:
            raise PolicyError(f"could not open policy {self.cfg.checkpoint}: {e}") from e

    def _encode(self, jpeg: bytes | None) -> str:
        """One base64 JPEG (no data: prefix) resized to the model's input size."""
        w, h = self.cfg.image_width, self.cfg.image_height
        if jpeg is None:
            if self._black is None:
                self._black = _jpeg(Image.new("RGB", (w, h)), self.cfg.jpeg_quality)
            jpeg = self._black
        else:
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            jpeg = jpeg if img.size == (w, h) else _jpeg(img.resize((w, h), Image.BILINEAR), self.cfg.jpeg_quality)
        return base64.b64encode(jpeg).decode()

    def observe(self, q: np.ndarray, jpegs: dict[str, bytes | None]) -> None:
        frames = {f"video.{slot}": self._encode(jpegs.get(slot)) for slot in self._slots}
        try:
            self._client.set_observation(frames, {**_canonical_state(q), "prompt": self._prompt},
                                         timestamp=time.monotonic())
        except self._dream.DreamError as e:
            raise PolicyError(f"observation rejected: {e}") from e

    def get_chunk(self) -> Trajectory:
        try:
            return chunk_to_canonical(self._client.get_action(), self.action_hz)
        except self._dream.DreamServerError as e:  # e.g. pool_warming while the model starts
            raise PolicyError(f"get_action refused: {e}", bool(e.retryable), e.retry_after_s) from e
        except self._dream.DreamError as e:
            raise PolicyError(f"get_action failed: {e}") from e

    def refill_due(self, executed: int, remaining: int) -> bool:
        return self._client.refill_due(executed, remaining)

    def health(self) -> tuple[bool, str]:
        try:
            models = {m.id: m for m in self._client.get_models()}
        except self._dream.DreamError as e:
            return False, f"dream API: {e}"
        m = models.get(self.cfg.checkpoint)
        if m is None:
            return False, f"checkpoint {self.cfg.checkpoint} not available to this key: {sorted(models)}"
        return m.servable, f"{m.id} servable={m.servable}, cameras {m.cameras}"

    def timing(self) -> dict | None:
        return self._client.last_timing

    def close(self) -> None:
        self._client.close()


def _jpeg(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class MockPolicyClient(PolicyClient):
    """Offline stand-in: sways the right arm and opens/closes its gripper around its starting pose.

    Exercises the whole loop (pacing, slew limiting, sim checks, supervisor) without the network.
    """

    name = "mock"

    def __init__(self, cfg: PolicyConfig, latency_s: float = 0.2, rows: int = 48):
        self.action_hz, self.latency_s, self.rows = cfg.action_hz, latency_s, rows
        self._q: np.ndarray | None = None
        self._anchor: np.ndarray | None = None  # pose of the first observation: the sway never drifts from it
        self._phase = 0.0
        self._last_get = 0.0
        self.observations = 0

    def open(self, prompt: str) -> None:
        self._prompt, self._phase, self._anchor = prompt, 0.0, None

    def observe(self, q: np.ndarray, jpegs: dict[str, bytes | None]) -> None:
        self._q = np.asarray(q, dtype=float).copy()
        if self._anchor is None:
            self._anchor = self._q.copy()
        self.observations += 1

    def get_chunk(self) -> Trajectory:
        if self._q is None:
            raise PolicyError("get_chunk before any observation")
        time.sleep(self.latency_s)
        t = self._phase + np.arange(self.rows) / self.action_hz
        out = np.tile(self._anchor, (self.rows, 1))
        out[:, 8] += 0.25 * np.sin(2 * np.pi * 0.25 * t)  # right joint1
        out[:, 15] = 0.5 + 0.4 * np.sin(2 * np.pi * 0.25 * t)  # right gripper
        self._phase += self.rows / self.action_hz
        return Trajectory(out, self.action_hz)

    def refill_due(self, executed: int, remaining: int) -> bool:
        return remaining <= int(self.latency_s * self.action_hz) + 5

    def health(self) -> tuple[bool, str]:
        return True, "mock policy (offline) — set policy.backend: dream for the real API"


def make_policy(cfg: PolicyConfig, api_key: str) -> PolicyClient:
    return DreamPolicyClient(cfg, api_key) if cfg.backend == "dream" else MockPolicyClient(cfg)


assert NUM_JOINTS == 16  # the 8-per-arm slicing above depends on the canonical layout
