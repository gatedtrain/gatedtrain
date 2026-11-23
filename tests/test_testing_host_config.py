"""Which `gatedtrain.testing` modules empty the host layout.

The test helpers (`gatedtrain.testing.support` and `gatedtrain.testing.ext`, which the recipe repositories' tests
import) make an unset `GATEDTRAIN_HOST_CONFIG` the empty file, so a test never reads the developer's host file. The
modules a distribution can register through entry points (the synthetic scenario pack, the stand-in and synthetic
adapters) must not: an installed `gatedtrain` command that loads them would otherwise ignore the user's host file.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

PRINT = "import os; print(os.environ.get('GATEDTRAIN_HOST_CONFIG'))"


def host_config_after(*modules: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "GATEDTRAIN_HOST_CONFIG"}
    code = "".join(f"import {m}; " for m in modules) + PRINT
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)
    return proc.stdout.strip()


class HostConfigDefaultTest(unittest.TestCase):
    def test_entry_point_modules_leave_the_host_file_alone(self):
        self.assertEqual(host_config_after("gatedtrain.testing.synthetic_pack", "gatedtrain.testing.standin",
                                           "gatedtrain.testing.synthetic"), "None")

    def test_test_helpers_use_the_empty_host_file(self):
        for module in ("gatedtrain.testing.support", "gatedtrain.testing.ext"):
            with self.subTest(module=module):
                self.assertEqual(host_config_after(module), os.devnull)


if __name__ == "__main__":
    unittest.main()
