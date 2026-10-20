"""DreamLab one-shot learning client.

Contract the rest of the pipeline relies on:
  plan(PlanRequest) -> Plan   (a joint-space Trajectory in canonical units)

Everything API-specific lives in HttpDreamLabClient._build_payload /
_parse_response. Fill those in from the DreamLab docs; nothing else changes.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx
import numpy as np

from orboh.config import DreamLabConfig
from orboh.demos.dataset import DemoEpisode
from orboh.types import JOINT_NAMES, Plan, Trajectory


@dataclass
class PlanRequest:
    task: str
    demo: DemoEpisode
    joints: np.ndarray  # current robot state, canonical (16,)
    images: dict[str, bytes] = field(default_factory=dict)  # camera role -> JPEG


class DreamLabError(RuntimeError):
    pass


class DreamLabClient(ABC):
    name = "dreamlab"

    @abstractmethod
    def plan(self, req: PlanRequest) -> Plan: ...

    def health(self) -> tuple[bool, str]:
        return True, "ok"


class MockDreamLabClient(DreamLabClient):
    """Stand-in until the real API is wired: replays the leader's demo motion."""

    name = "mock"

    def plan(self, req: PlanRequest) -> Plan:
        return Plan(Trajectory(req.demo.trajectory.positions.copy(), req.demo.trajectory.hz),
                    source="mock-replay", meta={"note": "demo replayed verbatim; no generalisation"})

    def health(self) -> tuple[bool, str]:
        return True, "mock backend (replays demo) — set dreamlab.backend: http for the real API"


class HttpDreamLabClient(DreamLabClient):
    name = "http"

    def __init__(self, cfg: DreamLabConfig):
        if not cfg.base_url or not cfg.api_key:
            raise DreamLabError("DREAMLAB_BASE_URL and DREAMLAB_API_KEY must be set in .env")
        self.cfg = cfg
        self._http = httpx.Client(base_url=cfg.base_url, timeout=cfg.timeout_s,
                                  headers={"Authorization": f"Bearer {cfg.api_key}"})

    def health(self) -> tuple[bool, str]:
        try:
            r = self._http.get(self.cfg.health_endpoint, timeout=10)
            return r.is_success, f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            return False, f"unreachable: {e}"

    def plan(self, req: PlanRequest) -> Plan:
        try:
            r = self._http.post(self.cfg.plan_endpoint, json=self._build_payload(req))
        except httpx.HTTPError as e:
            raise DreamLabError(f"request failed: {e}") from e
        if not r.is_success:
            raise DreamLabError(f"HTTP {r.status_code}: {r.text[:500]}")
        return self._parse_response(r.json())

    # ---- API-specific: VERIFY against DreamLab docs -------------------------
    def _build_payload(self, req: PlanRequest) -> dict:
        payload = {
            "task": req.task,
            "joint_names": list(JOINT_NAMES),
            "demo": {"fps": req.demo.trajectory.hz, "actions": req.demo.trajectory.positions.round(5).tolist(),
                     "task": req.demo.task},
            "observation": {"joints": req.joints.round(5).tolist()},
        }
        if self.cfg.send_images:
            payload["observation"]["images"] = {
                role: base64.b64encode(jpg).decode() for role, jpg in req.images.items()}
        return payload

    def _parse_response(self, body: dict) -> Plan:
        try:
            traj = Trajectory(np.asarray(body["trajectory"]["positions"], dtype=float),
                              float(body["trajectory"]["hz"]))
        except (KeyError, TypeError, ValueError) as e:
            raise DreamLabError(f"unexpected response shape: {e}; body keys={list(body)[:10]}") from e
        return Plan(traj, source="dreamlab", meta={k: v for k, v in body.items() if k != "trajectory"})


def make_dreamlab(cfg: DreamLabConfig) -> DreamLabClient:
    return HttpDreamLabClient(cfg) if cfg.backend == "http" else MockDreamLabClient()
