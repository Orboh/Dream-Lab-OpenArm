"""orboh <command>  — run `orboh -h` for the list."""

from __future__ import annotations

import argparse
import json
import sys

from orboh.config import load_config


def _pipeline(args):
    from orboh.pipeline import Pipeline

    cfg = load_config(args.config)
    if args.backend:
        cfg.robot.backend = args.backend
    p = Pipeline(cfg)
    p.subscribe(lambda e: e["type"] == "log" and e["level"] != "debug" and print(f"[{e['level']}] {e['msg']}"))
    p.start()
    return p


def cmd_serve(args):
    import uvicorn

    from orboh.pipeline import Pipeline
    from orboh.web.server import create_app

    cfg = load_config(args.config)
    if args.backend:
        cfg.robot.backend = args.backend
    if not cfg.web.token and cfg.web.host not in ("127.0.0.1", "localhost"):
        print("WARNING: ORBOH_WEB_TOKEN is empty and the console listens on the network. "
              "Anyone who can reach it can move the robot. Set it in .env.", file=sys.stderr)
    uvicorn.run(create_app(Pipeline(cfg)), host=args.host or cfg.web.host, port=args.port or cfg.web.port)


def cmd_preflight(args):
    p = _pipeline(args)
    p.preflight()
    for c in p.preflight_results:
        print(f"  {'OK  ' if c.passed else 'FAIL'} {c.name}: {c.detail}")
    p.shutdown()
    sys.exit(0 if p.steps["preflight"]["status"] == "ok" else 1)


def cmd_discover(args):
    p = _pipeline(args)
    p.discover_cameras(wiggle=args.wiggle)
    print(json.dumps(p.discovery.to_dict(), indent=2))
    p.shutdown()


def cmd_sample(args):
    p = _pipeline(args)
    p.make_sample_demos()
    p.shutdown()


def cmd_check(args):
    """Plan from a demo and run the safety check — never moves the robot."""
    p = _pipeline(args)
    p.make_plan(args.episode, args.task)
    p.check(args.speed)
    print(json.dumps(p.report.to_dict(), indent=2, default=str))
    p.shutdown()
    sys.exit(0 if p.report.passed else 1)


def cmd_run_policy(args):
    """Run the Dream-1 policy. Moves the robot unless --dry-run; real hardware also needs --approve."""
    cfg = load_config(args.config)
    if args.backend:
        cfg.robot.backend = args.backend
    if args.policy_backend:
        cfg.policy.backend = args.policy_backend
    from orboh.pipeline import GateError, Pipeline

    p = Pipeline(cfg)
    p.subscribe(lambda e: e["type"] == "log" and e["level"] != "debug" and print(f"[{e['level']}] {e['msg']}"))
    p.start()
    try:
        res = p.run_policy(args.prompt, args.seconds, approved=args.approve, dry_run=args.dry_run, speed=args.speed)
    except GateError as e:
        print(f"refused: {e}", file=sys.stderr)
        p.shutdown()
        sys.exit(2)
    print(json.dumps({k: v for k, v in res.__dict__.items() if k not in ("events", "timings")}, indent=2))
    p.shutdown()
    sys.exit(0 if res.completed else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="orboh", description="OpenArm + DreamLab one-shot manipulation pipeline")
    ap.add_argument("--config", help="path to pipeline.yaml")
    ap.add_argument("--backend", choices=["sim", "lerobot", "bridge"], help="override robot.backend")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="start the web control panel")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(fn=cmd_serve)
    sub.add_parser("preflight", help="run preflight checks").set_defaults(fn=cmd_preflight)
    s = sub.add_parser("discover-cameras", help="find RealSense cameras and assign center / right_wrist")
    s.add_argument("--wiggle", action="store_true", help="open/close the right gripper to identify the wrist camera")
    s.set_defaults(fn=cmd_discover)
    sub.add_parser("sample-demos", help="write a synthetic demo dataset for testing").set_defaults(fn=cmd_sample)
    s = sub.add_parser("check", help="plan from a demo and run the sim safety check (no robot motion)")
    s.add_argument("--episode", type=int, default=0)
    s.add_argument("--task", default="")
    s.add_argument("--speed", type=float, default=0.5)
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("run-policy", help="drive the arms with the Dream-1 policy (closed loop)")
    s.add_argument("--prompt", default="", help="task text, e.g. 'pick up the red cup'")
    s.add_argument("--seconds", type=float, help="run length (default policy.max_duration_s)")
    s.add_argument("--dry-run", action="store_true", help="query + safety-check chunks, never move the robot")
    s.add_argument("--approve", action="store_true", help="operator approval to move the arms")
    s.add_argument("--policy-backend", choices=["mock", "dream"], help="override policy.backend")
    s.add_argument("--speed", type=float, help="fraction of the safety speed limits, 0.1..1 (default policy.speed_scale 0.3)")
    s.set_defaults(fn=cmd_run_policy)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
