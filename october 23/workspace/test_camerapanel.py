"""Tests for the camera panel's worker (no cameras, no browser): on/off, frames delivered, errors shown instead of crashing."""

import threading
import time

import numpy as np
import pytest

import camerapanel as cp


class FakeSource:
    def __init__(self, fail_after=None):
        self.reads, self.closed, self.fail_after = 0, False, fail_after

    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise RuntimeError("Frame didn't arrive")
        return np.full((4, 6, 3), self.reads, dtype=np.uint8)

    def close(self):
        self.closed = True


def wait_for(condition, seconds=2.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def make_worker(source=None, opener=None, period=0.01):
    frames, status = [], []
    source = source or FakeSource()
    worker = cp.CameraWorker("head", opener or (lambda: source), frames.append, status.append, period=period)
    return worker, source, frames, status


def test_start_delivers_frames_and_stop_closes_the_camera():
    worker, source, frames, status = make_worker()
    worker.start()
    assert wait_for(lambda: len(frames) >= 3)
    assert worker.running
    worker.stop()
    assert source.closed and not worker.running
    assert frames[-1] is None  # the panel is cleared when the camera goes off
    assert status[-1] == "off"


def test_starting_twice_opens_one_camera():
    opened = []
    worker, source, _, _ = make_worker(opener=lambda: opened.append(1) or FakeSource())
    worker.start(); worker.start()
    assert wait_for(lambda: opened)
    worker.stop()
    assert len(opened) == 1


def test_a_camera_that_cannot_open_is_reported_not_raised():
    def broken():
        raise RuntimeError("No device connected")

    worker, _, frames, status = make_worker(opener=broken)
    worker.start()
    assert wait_for(lambda: any("cannot open" in s for s in status))
    assert not wait_for(lambda: worker.running, 0.2)
    assert wait_for(lambda: bool(frames) and frames[-1] is None)  # the panel is cleared right after the status text


def test_a_camera_that_fails_while_running_is_closed_and_reported():
    worker, source, frames, status = make_worker(source=FakeSource(fail_after=2))
    worker.start()
    assert wait_for(lambda: source.closed)
    assert any("stopped" in s for s in status)
    assert not wait_for(lambda: worker.running, 0.2)


def test_worker_can_be_restarted_after_stop():
    worker, source, frames, _ = make_worker()
    worker.start(); assert wait_for(lambda: frames); worker.stop()
    frames.clear()
    worker.start()
    assert wait_for(lambda: len(frames) >= 2)
    worker.stop()


def test_cameras_from_station_reads_realsense_serials(tmp_path):
    station = tmp_path / "station.yaml"
    station.write_text("cameras:\n- {name: head, serial: '344422070898', width: 1280}\n- {name: usb, device: 0}\n- {name: wrist_l, serial: '409122272149'}\n")
    assert cp.cameras_from_station(station) == [{"name": "head", "serial": "344422070898"}, {"name": "wrist_l", "serial": "409122272149"}]
    assert cp.cameras_from_station(tmp_path / "missing.yaml") == []
