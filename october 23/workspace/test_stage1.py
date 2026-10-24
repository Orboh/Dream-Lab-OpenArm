"""Tests for stage1.py: the station-log timeline, the inventory hash and the latency summary, on small invented samples."""

import json

import stage1

LOG = """=== dreamcli robot up: 2026-10-23 17:50:28 /x/python -m robots.station
2026-10-23 17:50:40,000 INFO _robots_driver_openarm_v2: arm left: locked (arms.py), so it stays where it is: not raised to ready
2026-10-23 17:50:41,000 INFO _robots_driver_openarm_v2: arm right: rest -> ready along 9 waypoints
2026-10-23 17:51:39,100 INFO __main__: station: idle -> starting (run 1, prompt 'move right arm 5 cm up')
2026-10-23 17:51:41,100 INFO __main__: station: starting -> running (run 1, prompt 'move right arm 5 cm up')
2026-10-23 17:51:50,885 INFO _robots_driver_openarm_v2: arm left: Dream asks: 57 deg from where this run began (joint 7), 155 deg of joint travel in all; this arm is LOCKED, so it holds still
2026-10-23 17:51:50,950 INFO robots.home: home: moving ['left', 'right'] home with grippers open over 3.0 s in 120 steps
2026-10-23 17:51:52,000 WARNING robots.cameras: camera 'wrist_r' source returned no frame; retrying for 2 s
2026-10-23 17:51:53,000 INFO _robots_driver_openarm_v2: arm right: parking at rest along the path before switching off
"""


def test_timeline_labels_each_movement_by_source_and_phase():
    (s,) = stage1.parse_station_log(LOG)
    labels = {(src, phase) for _, src, phase, _ in s["events"]}
    assert ("driver", "start: arm locked, not raised") in labels and ("driver", "start: raise to ready") in labels
    assert ("dream", "run: Dream's request") in labels and ("station", "reset/home: Dream's Home move") in labels and ("driver", "stop: park at rest") in labels
    assert s["runs"][1]["prompt"] == "move right arm 5 cm up" and [x[2] for x in s["runs"][1]["states"]] == ["starting", "running"]
    assert s["cameras"]["head/wrist: no frame"] == 1


def test_lines_before_the_first_station_start_are_ignored_and_each_start_is_its_own_session():
    sessions = stage1.parse_station_log("noise\n" + LOG + LOG)
    assert len(sessions) == 2 and "locked" in stage1.render_timeline(sessions)


def test_inventory_hashes_and_marks_a_missing_file(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("x")
    rows = stage1.inventory([a, tmp_path / "nope"])
    assert len(rows[0][0]) == 12 and rows[0][0] == stage1.sha12(a) and rows[1][0] == "missing"


def test_latency_counts_chunks_over_the_timeout_and_run_stops():
    t = "2026-10-23T04:52:27+00:00"
    rows = [{"kind": "inference", "t": t, "rtt_ms": ms, "gateway_ms": ms - 100, "network_ms": 70.0} for ms in (1000, 2000, 3500)]
    rows.append({"kind": "run_stop", "t": t, "run": 1, "chunks": 3, "stalls": 0, "clamps": 1, "reason": "ArmError"})
    out = stage1.latency(rows)
    assert out["all"]["median_ms"] == 2000 and out["all"]["over_timeout"] == 1 and out["runs"][0]["reason"] == "ArmError"
    json.dumps(out)
