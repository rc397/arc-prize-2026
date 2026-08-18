"""Wait for the 41-hour run to finish, then start Phase 0. One process, not a chained launch.

A chained waiter and its target die in the same Windows Update reboot, and then nothing fires and
nothing says so. If this process dies, nothing has started, and that is visible.

Usage: python rig_chain_phase0.py <PID_OF_BATCH08>
"""
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
LOG = os.path.join(BASE, "experiments", "phase0.log")


def alive(pid):
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259


def main(pid):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("[%s] waiting on pid %d before Phase 0\n" % (time.strftime("%m-%d %H:%M:%S"), pid))
    while alive(pid):
        time.sleep(60)
    r = subprocess.run([sys.executable, "-u",
                        os.path.join(BASE, "scripts", "rig_queue_phase0.py")],
                       cwd=BASE, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return r.returncode == 0


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1])) else 1)
