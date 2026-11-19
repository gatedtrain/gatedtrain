"""The B18 executor policy: how a container that runs model-written code is isolated (06 security).

Any recipe whose evaluation or training environment executes generated programs runs them in an executor container
with exactly these limits (a read-only root file system, a noexec tmpfs /tmp, CPU, memory, swap, pids, open files and
no core dumps) and only these environment names. A recipe's registry row names its own executor image and code.
"""

from __future__ import annotations

EXECUTOR_LIMITS = {"cpus": 8, "memory": "8g", "memory_swap": "8g", "pids": 256, "nofile": "1024:1024", "core": 0}
EXECUTOR_LIMIT_OPTIONS = ("--cpus", "8", "--memory", "8g", "--memory-swap", "8g", "--pids-limit", "256",
                          "--ulimit", "nofile=1024:1024", "--ulimit", "core=0")
EXECUTOR_TMPFS_OPTIONS = "rw,nosuid,nodev,noexec,size=4g"
EXECUTOR_ENV = ("HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED")
