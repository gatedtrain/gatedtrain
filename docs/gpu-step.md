# The GPU wrapper `gpu-step.sh`: interface

`gatedtrain ext-launch` never starts a process. It prints one command that runs the container through the
operator's GPU wrapper, `<host_work>/bin/gpu-step.sh`, and `ext-collect` later reads the wrapper's log to account the
GPU time of the run. gatedtrain does not ship this wrapper. The script of the host that ran the published recipes is
not published either: an operator writes their own to the interface below. Everything here is what the code reads
or prints (`src/gatedtrain/ext/launch.py`, `src/gatedtrain/ext/collect.py`); the rules are MULTIMODAL-SPEC MM-31 and
MM-36 and `tests/README.md` EXT-24, EXT-28, EXT-46, EXT-53, EXT-57, EXT-64, EXT-67, EXT-82 and EXT-105.

`host_work` is the directory named by the host file (`host_work`, or `GATEDTRAIN_EXT_HOST_WORK`; see `README.md`,
"Configuration"). It holds:

| Path | Written by | Read by gatedtrain |
|---|---|---|
| `<host_work>/bin/gpu-step.sh` | the operator | `ext-launch`, `ext-launch --recheck` and `ext-collect` read its text and hash it; nothing in gatedtrain runs it |
| `<host_work>/gpu-guard.log` | the wrapper | `ext-collect` (GPU seconds, first start) |
| `<host_work>/gpu-log.md` | the operator | `ext-launch` and `--recheck` (the running total) |

## Invocation

`ext-launch` prints (EXT-24):

```sh
GATEDTRAIN_GPU_CAP_H=1000 GATEDTRAIN_GPU_USED_H=<used> GATEDTRAIN_GPU_STEP_EST_H=<est> <host_work>/bin/gpu-step.sh <label> <docker run arguments...>
```

- `<label>`: the first argument, `<request label>-<12 hex>`. The request label matches
  `[A-Za-z0-9][A-Za-z0-9_.-]{0,99}`; the 12 hex digits are the start of the SHA-256 of the canonical launch record
  with the request label in place (EXT-46). The wrapper writes this label, unchanged, into its log.
- `<docker run arguments...>`: everything after the label is what follows `docker run`: `--rm`, then (for a GPU
  step) `--runtime=nvidia --gpus all`, the network, user, security, `-e` environment and `-v` mount options, the image
  and the container's argv, each token shell-quoted. The wrapper runs `docker run` with exactly these arguments on the
  Docker daemon that runs gatedtrain's containers (MM-31 names the context `gatedtrain`, `launch.DOCKER_CONTEXT`). A
  CPU-only step (`gpus: false`) has the same shape without `--runtime=nvidia --gpus all` (EXT-105). An adapter's
  executor sidecar script (EXT-199) gets a second shape: `--read-only`, tmpfs mounts and the adapter's fixed limit
  options, and no `--shm-size`.
- Environment, set on the command line:

  | Variable | Value |
  |---|---|
  | `GATEDTRAIN_GPU_CAP_H` | `1000` (`launch.GPU_CAP_H`), a GPU-hour budget ceiling high enough not to limit runs |
  | `GATEDTRAIN_GPU_USED_H` | the record's `gpu_used_h`: the last running total of `gpu-log.md` when `ext-launch` ran |
  | `GATEDTRAIN_GPU_STEP_EST_H` | the request's `gpu_est_h` for a smoke run, the PASS smoke record's `projected_gpu_h` for a full run (EXT-27) |

  gatedtrain passes these values and does not check what the wrapper does with them. The intended use is that the
  wrapper refuses a step whose used hours plus estimate would pass the cap.
- Exit status: the wrapper exits with the container's exit status after it logs `END ... rc=<rc>` (EXT-46).

`ext-launch --recheck` right before the run requires `GATEDTRAIN_GPU_USED_H` to still equal the running total of
`gpu-log.md` (EXT-82), so log a step's hours after it ends, not between the recheck and the run.

## What gatedtrain reads in the script

`ext-launch` and `ext-collect` read the script as text (`launch.gpu_step_log_problem`). They do not run it, and the
check guards against a misconfigured `host_work`, not against a forged script (EXT-57). The script must have:

- exactly one line that is, in full, `WORK=<path>` (the path may be quoted with `"` or `'`), where `<path>` is
  `host_work`;
- a line that is exactly `LOG="$WORK/gpu-guard.log"` (surrounding whitespace aside);
- no other assignment to `WORK` or `LOG` anywhere in a line (`X=`, `export X=`, `a; X=`, `${X:=...}`, `${X=...}`,
  `X+=`); uses such as `$LOG` or `"$WORK/..."` are not assignments. Lines that start with `#` are ignored.

Together these make `<host_work>/gpu-guard.log` the file the wrapper writes (EXT-46). A script that fails the check
is refused at `ext-launch` (`EXT-GPU --request.host_work`) and at `ext-collect` (`EXT-GPU --launch-record.host_work`).

**Hash pin.** `ext-launch` records the SHA-256 of the script's bytes as `gpu_step_sha256` in the launch record.
`ext-launch --recheck` and `ext-collect` hash the script again and refuse when it differs
(`EXT-GPU ...gpu_step_sha256`, EXT-67): the wrapper that accounted the GPU time must be the one that was checked.
Do not edit the script between a launch and its collection.

## `gpu-guard.log`

One event per line, whitespace-separated tokens:

```text
<UTC time> <EVENT> <label> [<key>=<value> ...]
```

The timestamp uses UTC with the format `%Y-%m-%dT%H:%M:%SZ`. `ext-collect` reads only lines whose second
token is `START`, `END` or `PREEMPTED` and whose third token is exactly the launch label; every other line (any
other event, or another label) is ignored, so the wrapper may log more (the test fixture
`tests/fixtures/ext/gpu-guard.fixture.log` has `GUARD`, `WAIT`, `PREEMPT` and `REFUSED` lines).

| Event | When | Required fields |
|---|---|---|
| `START` | right before the container starts | none (`attempt=<n>` is customary) |
| `END` | after the container exits on its own | `rc=<int>`, exactly once: the container's exit status |
| `PREEMPTED` | after the container was stopped from outside, for example to free the GPU | none (`rc=<int>` is customary) |

A wrapper may retry a preempted or failed attempt under the same label. `ext-collect` (`launch.gpu_seconds`):

- pairs each `START` of the label with the next `END` or `PREEMPTED` of the label and sums the pair durations over
  all attempts, preempted ones included (EXT-28); the sum is the run's `wall_seconds`, and its `gpu_seconds` for a
  GPU step (0 for a CPU-only step, EXT-105);
- refuses (`EXT-GPU --launch-record.label`) when the label is absent, a timestamp does not parse, a `START` follows an
  unterminated `START`, an `END` or `PREEMPTED` has no `START`, the last `START` has no terminator, the label has no
  `END`, an `END` lacks a single `rc=<int>`, the label's last terminal event is not an `END` (EXT-53), or that last
  `END` has an rc other than 0 (EXT-46);
- requires the files it reads from the run (the stage-out's fixed output files, every file of the cut directories, and
  the logs the adapter names) to be modified at or after the label's first `START` (EXT-64), so log `START` before the
  container starts.

Example (from the test fixture; label `ext-synthetic-full`, without the hash suffix):

```text
2026-11-11T09:00:00Z START ext-synthetic-full attempt=1
2026-11-11T09:05:02Z PREEMPTED ext-synthetic-full attempt=1 rc=137
2026-11-11T09:30:00Z START ext-synthetic-full attempt=2
2026-11-11T10:00:00Z END ext-synthetic-full attempt=2 rc=1
2026-11-11T10:10:00Z START ext-synthetic-full attempt=1
2026-11-11T10:40:00Z END ext-synthetic-full attempt=1 rc=0
```

## `gpu-log.md`

The operator's GPU-time log, one row per GPU step, a Markdown table (as in `docs/tier-b-runbook.md` section 1 and
`tests/fixtures/ext/gpu-log.fixture.md`):

```markdown
| Step | Started (UTC) | Ended (UTC) | GPU-hours | Running total | Outcome |
|---|---|---|---|---|---|
| 7 long run, attempt 2 | 2026-11-11T09:17:49Z | 2026-11-11T10:30:47Z | 1.22 | 2.69 | PASS rc=0 |
```

gatedtrain reads only the `Running total` column (`launch.gpu_used_h`): a table starts at a `|` row with a cell
`Running total`, the separator row is skipped, and the table ends at the first line that does not start with `|`.
The value used is the running total of the last data row of any such table; it must parse as a number. A file
without such a row, or with a running total that is not a number, is refused. `ext-launch --gpu-log` must name
`<host_work>/gpu-log.md` (EXT-67). Other columns, headings and notes are free text.
