"""Test helpers for gatedtrain and for the recipe repositories' adapter tests.

`gatedtrain.testing.support`: an independent implementation of the canonical JSON and manifest hashes, and running
the `gatedtrain` CLI in a subprocess and parsing its errors. `gatedtrain.testing.ext`: fixture builders for external runs (launch
labels, the host-layout hash, the gpu-step.sh stand-in). They deliberately do not call `gatedtrain.canonical`, so a test
checks gatedtrain's hashes against a second implementation. `gatedtrain.testing.synthetic`: a synthetic external-run adapter,
registered only on request (`install()`, or `python -m gatedtrain.testing.synthetic <gatedtrain arguments>`), on which gatedtrain's
own external-run tests run, and an example of the adapter interface for adapter authors.

Importing the test helpers (`support`, `ext`) makes an unset `GATEDTRAIN_HOST_CONFIG` the empty host file; importing
this package or the modules registered through entry points does not.
"""
