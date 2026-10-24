"""Stage 1 of goal.md, read-only: what was really running, and what moved when and why. Nothing here opens a CAN bus, a camera or a motor.

`inventory` hashes the files in force; `timeline` turns `~/.dreamlabs/station.log` into runs with every movement labelled by phase and source;
`latency` summarises `~/robots-inference.jsonl`. Run from this folder: `python stage1.py inventory|timeline|latency`.
"""

import collections
import hashlib
import json
import pathlib
import re
import statistics
import sys

HOME = pathlib.Path("~").expanduser()
STATION_LOG = HOME / ".dreamlabs" / "station.log"
INFERENCE_LOG = HOME / "robots-inference.jsonl"
CHUNK_TIMEOUT_S = 3.0  # the station's default action_timeout_s (log-016)

TRANSITION = re.compile(r"^(\S+ \S+) \w+ __main__: station: (\w+) -> (\w+) \(run (\d+), prompt '(.*?)'?\)?$")
STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ ")
# (regex, source, phase): the first match labels the line. Source says who moved the arm; phase says when in the station's life.
RULES = [
    (re.compile(r"arm (\w+): rest -> ready along"), "driver", "start: raise to ready"),
    (re.compile(r"arm (\w+): locked \(arms\.py\)"), "driver", "start: arm locked, not raised"),
    (re.compile(r"arm (\w+): parking at rest"), "driver", "stop: park at rest"),
    (re.compile(r"arm (\w+): [\d.]+ deg from the rest-to-ready path, not parking"), "driver", "stop: off path, not parked"),
    (re.compile(r"arm (\w+): Dream asks: (.*)"), "dream", "run: Dream's request"),
    (re.compile(r"arm (\w+): Dream run ended: (.*)"), "dream", "run: ended"),
    (re.compile(r"arm (\w+): stopped at the edge of the allowed space"), "guard", "run: cut at the zone edge"),
    (re.compile(r"home: moving \[(.*?)\] home"), "station", "reset/home: Dream's Home move"),
    (re.compile(r"home: arm '(\w+)' write failed"), "station", "reset/home: write failed"),
    (re.compile(r"home: arms .* at home and holding"), "station", "reset/home: reached"),
    (re.compile(r"arm (\w+): smooth move of ([\d.]+) s"), "driver", "move: smooth tracker"),
]


def sha12(path: pathlib.Path) -> str:
    """First 12 hex digits of the file's sha256 ('missing' when absent)."""
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return "missing"


def inventory(paths: list[pathlib.Path]) -> list[tuple[str, str, str]]:
    """`(hash, modified time, path)` per file, so two sessions can be told apart by what was in force."""
    import datetime as dt

    rows = []
    for path in map(pathlib.Path, paths):
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%m-%d %H:%M") if path.exists() else "-"
        rows.append((sha12(path), mtime, str(path)))
    return rows


def parse_station_log(text: str) -> list[dict]:
    """Splits the log at each `dreamcli robot up` and lists, per start, the labelled events and the run states.

    @return One dict per station start: `{"started", "events": [(time, source, phase, detail)], "runs": {run: {prompt, states: [(time, from, to)]}}, "cameras": counts}`.
    """
    sessions: list[dict] = []
    for line in text.splitlines():
        if line.startswith("=== dreamcli robot up:"):
            sessions.append({"started": line.split("up:", 1)[1].strip().split(" ")[0:2], "events": [], "runs": {}, "cameras": collections.Counter()})
            continue
        if not sessions:
            continue
        s, stamp = sessions[-1], STAMP.match(line)
        if not stamp:
            continue
        t = stamp.group(1)
        moved = TRANSITION.match(line)
        if moved:
            _, src, dst, run, prompt = moved.groups()
            s["runs"].setdefault(int(run), {"prompt": prompt, "states": []})["states"].append((t[11:], src, dst))
            continue
        for key, label in (("no frame", "head/wrist: no frame"), ("is back at", "camera back"), ("older than stale_after_s", "stale frame")):
            if key in line:
                s["cameras"][label] += 1
        for rule, source, phase in RULES:
            match = rule.search(line)
            if match:
                s["events"].append((t[11:], source, phase, " ".join(g for g in match.groups() if g)[:110]))
                break
    return sessions


def render_timeline(sessions: list[dict]) -> str:
    """A compact text view: per station start, run states then the movement events grouped by phase."""
    out = []
    for i, s in enumerate(sessions, 1):
        if not s["events"] and not s["runs"]:
            continue
        out.append(f"## station start {i} at {' '.join(s['started'])}")
        for run, info in s["runs"].items():
            states = " ".join(f"{a}>{b}@{t[:8]}" for t, a, b in info["states"][:14])
            out.append(f"- run {run} '{info['prompt'][:60]}': {states}")
        count = collections.Counter((src, phase) for _, src, phase, _ in s["events"])
        out.append("- movement events: " + "; ".join(f"{src}/{phase} x{n}" for (src, phase), n in sorted(count.items())))
        if s["cameras"]:
            out.append("- camera notes: " + ", ".join(f"{k} x{v}" for k, v in s["cameras"].items()))
    return "\n".join(out)


def latency(rows: list[dict]) -> dict:
    """Round-trip numbers from the inference log: overall, per local hour, and per run (chunks, stalls, clamps, why it stopped)."""
    inf = [r for r in rows if r.get("kind") == "inference" and r.get("rtt_ms") is not None]

    def stats(sample: list[dict]) -> dict:
        rtt = sorted(r["rtt_ms"] for r in sample)
        return {"n": len(rtt), "median_ms": round(statistics.median(rtt)), "p90_ms": round(rtt[int(0.9 * (len(rtt) - 1))]), "max_ms": round(rtt[-1]),
                "gateway_median_ms": round(statistics.median(r["gateway_ms"] for r in sample)), "network_median_ms": round(statistics.median(r["network_ms"] for r in sample)),
                "over_timeout": sum(1 for r in sample if r["rtt_ms"] > CHUNK_TIMEOUT_S * 1000)} if rtt else {"n": 0}

    import datetime as dt

    by_hour: dict[str, list[dict]] = collections.defaultdict(list)
    for r in inf:
        by_hour[dt.datetime.fromisoformat(r["t"]).astimezone().strftime("%m-%d %H:00")].append(r)
    runs = [{"run": r["run"], "t": r["t"], "chunks": r.get("chunks"), "stalls": r.get("stalls"), "clamps": r.get("clamps"), "reason": r.get("reason")} for r in rows if r.get("kind") == "run_stop"]
    return {"all": stats(inf), "by_hour": {h: stats(v) for h, v in sorted(by_hour.items())}, "runs": runs}


def main(argv: list[str]) -> int:
    command = argv[0] if argv else "timeline"
    if command == "inventory":
        paths = [HOME / ".dreamlabs" / n for n in ("station.yaml", "zones.yaml", "active_arms.json", "arms.lock")]
        paths += sorted(pathlib.Path("zones").glob("zones-*.yaml"))[-2:] + [pathlib.Path(p) for p in ("../urdf/openarm_bimanual.urdf", "../driver/openarm_v2.py", "../driver/openarm_guard.py", "../driver/arms.py")]
        for digest, mtime, path in inventory(paths):
            print(digest, mtime, path)
    elif command == "timeline":
        print(render_timeline(parse_station_log(STATION_LOG.read_text(encoding="utf-8", errors="replace"))))
    elif command == "latency":
        rows = [json.loads(line) for line in INFERENCE_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(json.dumps(latency(rows), indent=1))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
