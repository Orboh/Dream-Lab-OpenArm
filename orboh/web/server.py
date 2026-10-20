"""FastAPI control panel: REST for actions, WebSocket for live state/logs, MJPEG for cameras."""

from __future__ import annotations

import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from orboh.pipeline import Busy, GateError, Pipeline

STATIC = Path(__file__).parent / "static"


class DiscoverBody(BaseModel):
    wiggle: bool = False


class RolesBody(BaseModel):
    assignments: dict[str, str]


class RecordBody(BaseModel):
    mode: str = "dry"


class PlanBody(BaseModel):
    episode: int
    task: str = ""


class CheckBody(BaseModel):
    speed_scale: float = 1.0


class ExecuteBody(BaseModel):
    approved: bool = False


class PolicyBody(BaseModel):
    prompt: str = ""
    seconds: float | None = None
    dry_run: bool = True  # the safe default: query Dream and check its chunks, move nothing
    approved: bool = False
    backend: str | None = None  # mock | dream; default from policy.backend
    speed: float | None = None  # fraction of the safety speed limits, 0.1..1 (default policy.speed_scale)
    gripper: str | None = None  # "model" (Dream decides) | "demos" (open/close amounts from the recorded demos); default from policy.gripper_mode
    mode: str = "robot"  # "robot" (the real arms, gated) | "sim" (MuJoCo twin only: nothing real moves, no gates)


def create_app(pipeline: Pipeline) -> FastAPI:
    token = pipeline.cfg.web.token
    sockets: set[WebSocket] = set()
    loop_holder: dict = {}

    def authorized(supplied: str | None) -> bool:
        return not token or (supplied is not None and hmac.compare_digest(supplied, token))

    def require_token(request: Request) -> None:
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ") or request.query_params.get("token")
        if not authorized(supplied):
            raise HTTPException(401, "missing or wrong ORBOH_WEB_TOKEN")

    def forward(event: dict) -> None:
        loop = loop_holder.get("loop")
        if loop:
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(broadcast(event)))

    async def broadcast(event: dict) -> None:
        if event["type"] == "state":
            event = {"type": "state", "state": pipeline.state()}
        msg = json.dumps(event, default=str)
        for ws in list(sockets):
            try:
                await ws.send_text(msg)
            except Exception:
                sockets.discard(ws)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop_holder["loop"] = asyncio.get_running_loop()
        pipeline.subscribe(forward)
        pipeline.start()
        yield
        pipeline.shutdown()

    app = FastAPI(title="Orboh OpenArm pipeline", lifespan=lifespan)
    auth = [Depends(require_token)]

    def task(name: str, fn) -> dict:
        try:
            pipeline.run_task(name, fn)
        except Busy as e:
            raise HTTPException(409, str(e))
        return {"started": name}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/simple")
    def simple():
        return FileResponse(STATIC / "simple.html")

    @app.get("/api/state", dependencies=auth)
    def state():
        return pipeline.state()

    @app.get("/api/logs", dependencies=auth)
    def logs():
        return pipeline.logs()

    @app.post("/api/estop", dependencies=auth)
    def estop():
        pipeline.estop()
        return {"estopped": True}

    @app.post("/api/estop/reset", dependencies=auth)
    def estop_reset():
        pipeline.stop_event.clear()
        pipeline.log("E-STOP released")
        return {"estopped": False}

    @app.post("/api/preflight", dependencies=auth)
    def preflight():
        return task("preflight", pipeline.preflight)

    @app.post("/api/cameras/discover", dependencies=auth)
    def discover(body: DiscoverBody):
        return task("camera discovery", lambda: pipeline.discover_cameras(body.wiggle))

    @app.post("/api/cameras/roles", dependencies=auth)
    def roles(body: RolesBody):
        return task("set camera roles", lambda: pipeline.set_camera_roles(body.assignments))

    @app.get("/api/cameras/{role}/snapshot.jpg", dependencies=auth)
    def snapshot(role: str):
        jpg = pipeline.cameras.snapshot(role)
        if jpg is None:
            raise HTTPException(404, f"no frame for camera role '{role}'")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{role}/stream", dependencies=auth)
    async def stream(role: str, fps: float = 10):
        async def frames():
            while True:
                jpg = await asyncio.to_thread(pipeline.cameras.snapshot, role)
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(1 / max(1.0, min(fps, 30.0)))
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/demos", dependencies=auth)
    def demos():
        return pipeline.list_demos()

    @app.post("/api/demos/sample", dependencies=auth)
    def sample():
        return task("generate synthetic demos", pipeline.make_sample_demos)

    @app.post("/api/demos/record", dependencies=auth)
    def record(body: RecordBody):
        return task(f"teleop recording ({body.mode})", lambda: pipeline.record_demo(body.mode))

    @app.post("/api/plan", dependencies=auth)
    def plan(body: PlanBody):
        return task("DreamLab plan", lambda: pipeline.make_plan(body.episode, body.task))

    @app.post("/api/check", dependencies=auth)
    def check(body: CheckBody):
        return task("sim safety check", lambda: pipeline.check(body.speed_scale))

    @app.get("/api/trajectory", dependencies=auth)
    def trajectory():
        return pipeline.trajectory_data()

    @app.get("/api/sim/frame/{i}.jpg", dependencies=auth)
    def sim_frame(i: int):
        jpg = pipeline.sim_frame(i)
        if jpg is None:
            raise HTTPException(404, "no such preview frame")
        return Response(jpg, media_type="image/jpeg")

    @app.post("/api/execute", dependencies=auth)
    def execute(body: ExecuteBody):
        try:
            if pipeline.cfg.safety.require_manual_approval and not body.approved:
                raise GateError("execution needs explicit operator approval")
            pipeline.gate()  # fail fast with a clear message before starting the task
        except GateError as e:
            raise HTTPException(412, str(e))
        return task("execute", lambda: pipeline.execute(body.approved))

    @app.post("/api/policy/run", dependencies=auth)
    def policy_run(body: PolicyBody):
        if body.gripper not in (None, "model", "demos"):
            raise HTTPException(422, f"unknown gripper mode '{body.gripper}' (model | demos)")
        if body.mode == "sim":
            backend = body.backend or pipeline.cfg.policy.backend
            if backend not in ("mock", "dream"):
                raise HTTPException(422, f"unknown policy backend '{backend}'")
            return task("Dream-1 simulation", lambda: pipeline.run_policy(body.prompt, body.seconds, backend=backend, simulate=True,
                                                                          gripper=body.gripper, speed=body.speed))
        if body.mode != "robot":
            raise HTTPException(422, f"unknown mode '{body.mode}' (robot | sim)")
        try:
            prompt, backend = pipeline.policy_gate(body.prompt, body.backend, body.dry_run, body.approved, need_dry_first=True, gripper=body.gripper)
        except GateError as e:
            raise HTTPException(412, str(e))
        if backend not in ("mock", "dream"):
            raise HTTPException(422, f"unknown policy backend '{backend}'")
        label = "Dream-1 dry-run" if body.dry_run else "Dream-1 run"
        return task(label, lambda: pipeline.run_policy(prompt, body.seconds, approved=body.approved,
                                                       dry_run=body.dry_run, backend=backend, need_dry_first=True, gripper=body.gripper, speed=body.speed))

    @app.get("/api/sim/live.jpg", dependencies=auth)
    def sim_live():
        jpg = pipeline.sim_live_jpeg()
        if jpg is None:
            raise HTTPException(404, "no simulation run yet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/policy/snapshot/{slot}.jpg", dependencies=auth)
    def policy_snapshot(slot: str):
        jpg = pipeline.snapshot_frames.get(slot)
        if jpg is None:
            raise HTTPException(404, f"no snapshot frame for camera slot '{slot}'")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/policy/chunks", dependencies=auth)
    def policy_chunks():
        return pipeline.policy_chunks

    @app.get("/api/detect", dependencies=auth)
    def detect():
        try:
            return [d.to_dict() for d in pipeline.find_block()]
        except GateError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/home", dependencies=auth)
    def home():
        return task("go home", pipeline.go_home)

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        if not authorized(socket.query_params.get("token")):
            await socket.close(code=4401)
            return
        await socket.accept()
        sockets.add(socket)
        await socket.send_text(json.dumps({"type": "state", "state": pipeline.state()}, default=str))
        try:
            while True:
                await socket.receive_text()
        except WebSocketDisconnect:
            sockets.discard(socket)

    return app
