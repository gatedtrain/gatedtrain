# Development

This page is for people who change gatedtrain or run its tests. It shows how to run the tests, lists the design
references the code and tests cite, and says where the unpublished references and recipe paths stand.

## Tests

From the repository root, with the package and its `tier-a` extra installed in `.venv` (the tests run the CLI with
the interpreter that runs the suite):

```sh
.venv/bin/python -m unittest discover -s tests -t tests
```

The full suite takes about 13 minutes on a CPU; `.github/workflows/test.yml` runs it on Python 3.12 and 3.13 for
every push and pull request, with the design set at `rev-28`. Some tests read external sources and skip, naming the
variable, when it is unset:

| Variable | Contents |
|---|---|
| `GATEDTRAIN_DESIGN_DIR` | the design set `gatedtrain-design` |
| `GATEDTRAIN_NEMO_EVALUATOR_DIR` | a NeMo Evaluator clone at `ea35ba36756abd59de201f60af8a2feba014744d` |
| `GATEDTRAIN_NEMO_RL_DIR` | a NeMo RL clone at `4f39fd86289a2ffd5aaf5d202702f7fd7185c924` |
| `GATEDTRAIN_TEST_TIERB_ROOT` | the Tier B host work directory of a host that ran the Tier B run (`bin/gpu-step.sh`, `tierb/`, `runs/`), read only |
| `GATEDTRAIN_TEST_MM_ROOT` | the runtime and output root of a host that ran the multimodal recipes, read only (for the recipes' tests) |

`GATEDTRAIN_SKIP_SLOW=1` skips the slow end-to-end runs. The suite uses a synthetic adapter. See the
[test reference](../tests/README.md) for coverage and fixture details.

## Documentation site

The pages under `docs/`, with `README.md` as the home page and `CHANGELOG.md`, are also published as a website by
MkDocs (`mkdocs.yml`; `mkdocs_hooks.py` adds the two root files and rewrites links written for GitHub). Every pull
request builds it with `mkdocs build --strict`, so a broken link fails the check; main publishes it to GitHub Pages.
To preview it locally:

```sh
python -m pip install -r requirements-docs.txt
mkdocs serve
```

## Design references

This version implements revision 28 of [gatedtrain-design](https://github.com/gatedtrain/gatedtrain-design), except
the training-seed unit of revision 20 (05 §3.3).
The local specifications cover the [CPU demo](../MVP-SPEC.md), [Tier B helpers](../TIER-B-SPEC.md) and
[external adapters](../MULTIMODAL-SPEC.md). The [findings log](findings.md) records design questions and
regression fixes.

### Unpublished references

Some specifications and test notes cite development decisions, interim rules and review findings by number.
[Section 9 of the test reference](../tests/README.md#9-design-gaps-found-while-writing-the-tests) lists G-1 through G-11.

Paths under `<mm-root>` refer to the host layout used for the recipe runs. The matching instructions are in
`runtime/README.md` in each [recipe repository](external-runs.md#recipe-repositories).

| Host directory | Request area | Recipe |
|---|---|---|
| `area1-trl` | 1 | TRL GRPO on GSM8K |
| `area2-drones` | 4 | Stable-Baselines3 PPO |
| `area3-sd15` | 2, row 2a | DDPO with SD1.5 |
| `area3-video` | 3 | Flow-GRPO with Wan |
| `area4-whisper` | 5 | Whisper SFT on AMI |
| `area5-sd35` | 2, row 2b | Flow-GRPO with SD3.5 |
| `area6-cad` | 6 | NeMo RL for CAD |
| `area7-vlm` | 7 | NeMo RL on CLEVR |
| `area8-vla` | 8 | LeRobot SmolVLA on LIBERO |
| `area10-agentic` | 10 | NeMo RL for agentic math |
| `area11-tts` | 11 | CosyVoice DPO |
