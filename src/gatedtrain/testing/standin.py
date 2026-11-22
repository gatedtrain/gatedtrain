"""A stand-in ext adapter for the AI-native local backend (AN-22), for gatedtrain's own tests.

The local backend applies the launch rules of an ext adapter to every job (`backend.job_problems`). This adapter's
rules fit the jobs the AI-native tests run: read-only inputs from directories under the host's output root (the
backend's `new_output_dir`), mounted under `/inputs`; the output at `/out`; a `/tmp` tmpfs; scripts run by the image's
own interpreter (`/usr/local/bin/python3`), so no interpreter is mounted; the environment names `HF_HUB_OFFLINE` and
`HOME` (a path under `/tmp` or `/out`); no sidecar. Its run-reading methods are the synthetic adapter's
(`gatedtrain.testing.synthetic`), whose evaluator it names, so it conforms (`gatedtrain.ext.conformance`).

It is never registered by an installed distribution of gatedtrain: a test registers it with `install()`.
"""

from __future__ import annotations

from typing import Any

from . import synthetic

NAME = "ainative-standin"
TRAINER_ID = "ainative.standin@0.1"
FRAMEWORK_PACKAGE = "gatedtrain-standin"
IMAGE_PYTHON = "/usr/local/bin/python3"

TRAINER_ROWS: dict[str, dict[str, Any]] = {NAME: {
    "uses": (TRAINER_ID,),
    "pins": (FRAMEWORK_PACKAGE,),
    "outputs": ("the job's files under /out",),
    "components": synthetic.HF_LLM_COMPONENTS,
    "consumption": {"target": "stage", "fallback": None},
    "gpu_h_estimate": 0.0,
    "evaluators": (synthetic.EVALUATOR_ID,),
    "framework": FRAMEWORK_PACKAGE,
    "input_roots": ("{output_root}",),
    "spec_binding": False,
    "env_names": ("HF_HUB_OFFLINE", "HOME"),
    "scan_skip": (),
    "interpreter_trees": (),
    "interpreter": IMAGE_PYTHON,
    "container_paths": {"ro_identity": False, "ro_under": ["/inputs"], "rw": ["/out"], "tmpfs": ["/tmp"]},
    "env_paths": ("HOME",),
    "image_interpreters": {"synthetic_train_job.py": IMAGE_PYTHON, "synthetic_batch_generate.py": IMAGE_PYTHON},
}}

PLUGIN = {"adapters": {NAME: "gatedtrain.testing.standin:StandinAdapter"}, "trainer_rows": TRAINER_ROWS}


def install() -> None:
    """Register the stand-in adapter (and the synthetic adapter, whose evaluator it names) in this process
    (idempotent)."""
    from .. import spec
    from ..ext import registry

    synthetic.install()
    if registry.ADAPTERS.get(NAME) != PLUGIN["adapters"][NAME]:
        registry.register(PLUGIN, "gatedtrain.testing.standin.PLUGIN")
    spec._external_ids()


class StandinAdapter(synthetic.SyntheticAdapter):
    """The synthetic adapter's interface under the stand-in's name, trainer id and framework."""

    name = NAME
    trainer_id = TRAINER_ID
    framework_name = FRAMEWORK_PACKAGE
    binds_spec = False
