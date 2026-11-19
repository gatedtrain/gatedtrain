"""Test helpers for gatedtrain and for the recipe repositories' adapter tests.

`gatedtrain.testing.support`: an independent implementation of the canonical JSON and manifest hashes, and running
the `gatedtrain` CLI in a subprocess and parsing its errors. `gatedtrain.testing.ext`: fixture builders for external runs (launch
labels, the host-layout hash, the gpu-step.sh stand-in). They deliberately do not call `gatedtrain.canonical`, so a test
checks gatedtrain's hashes against a second implementation. `gatedtrain.testing.synthetic`: a synthetic external-run adapter,
registered only on request (`install()`, or `python -m gatedtrain.testing.synthetic <gatedtrain arguments>`), on which gatedtrain's
own external-run tests run, and an example of the adapter interface for adapter authors.
"""

import os

# A test never reads the developer's host file (`~/.config/gatedtrain/host.toml`): its daemon lists enter
# `host_config_sha256`, so the layouts the tests build would not match. Unless the environment names a host file, an
# empty one applies, in process and in the CLI subprocesses the tests start.
os.environ.setdefault("GATEDTRAIN_HOST_CONFIG", os.devnull)
