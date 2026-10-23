"""Track ContamX / EnergyPlus child processes so a Lab cancel can stop them."""
from __future__ import annotations

import os
import subprocess
import threading

_LOCK = threading.Lock()
_PROCS: list[subprocess.Popen] = []


def _register(proc: subprocess.Popen) -> None:
    with _LOCK:
        _PROCS.append(proc)


def _unregister(proc: subprocess.Popen) -> None:
    with _LOCK:
        if proc in _PROCS:
            _PROCS.remove(proc)


def kill_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=8,
            )
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def cancel_active_solvers() -> int:
    with _LOCK:
        procs = list(_PROCS)
    for proc in procs:
        kill_proc(proc)
    return len(procs)


def run_solver(cmd: list[str], *, timeout_s: float, cwd: str | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _register(proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        kill_proc(proc)
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(cmd, timeout_s, output=stdout, stderr=stderr)
    finally:
        _unregister(proc)
