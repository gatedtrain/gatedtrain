# Third-party notices

gatedtrain contains no ported third-party code apart from one prompt string and one function of Python's standard
library (below). It redistributes a small amount of third-party data in test fixtures.

## Python's `posixpath.realpath`

- **What:** `src/gatedtrain/paths.py` (`realpath`) is Python 3.13's `posixpath.realpath` for `str` paths in its
  non-strict mode, ported line for line, so that lenient path resolution is the same on Python 3.12 and 3.13.
- **Source:** CPython 3.13.18, `Lib/posixpath.py`, function `realpath`. Copyright (c) 2001-2024 Python Software
  Foundation; All Rights Reserved.
- **Changes (PSF License, clause 3):** `str` paths only (no `bytes`); the non-strict mode only (no `strict`,
  `ALLOW_MISSING` or link limit); names and comments shortened.
- **License:** Python Software Foundation License Version 2:

```
PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
--------------------------------------------

1. This LICENSE AGREEMENT is between the Python Software Foundation
("PSF"), and the Individual or Organization ("Licensee") accessing and
otherwise using this software ("Python") in source or binary form and
its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
analyze, test, perform and/or display publicly, prepare derivative works,
distribute, and otherwise use Python alone or in any derivative version,
provided, however, that PSF's License Agreement and PSF's notice of copyright,
i.e., "Copyright (c) 2001-2024 Python Software Foundation; All Rights Reserved"
are retained in Python alone or in any derivative version prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on
or incorporates Python or any part thereof, and wants to make
the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to Python.

4. PSF is making Python available to Licensee on an "AS IS"
basis.  PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON
FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS
A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON,
OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between PSF and
Licensee.  This License Agreement does not grant permission to use PSF
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using Python, Licensee
agrees to be bound by the terms and conditions of this License
Agreement.
```

## NeMo Evaluator GSM8K prompt

- **What:** `src/gatedtrain/tierb.py` (`NEL_GSM8K_PROMPT`) and `tests/_tierb.py` hold NeMo Evaluator's GSM8K prompt
  text, with its `{question}` placeholder written `{}`, so that the training prompt equals the evaluation prompt.
- **Source:** NeMo Evaluator, <https://github.com/NVIDIA-NeMo/Evaluator>, `src/nemo_evaluator/benchmarks/gsm8k.py`
  lines 22-26 at commit `ea35ba36`. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
- **License:** Apache-2.0 (the same license as gatedtrain, text in `LICENSE`). The project has no NOTICE file at
  that commit.


## GSM8K

- **What:** `tests/fixtures/r11/longrun/raw_samples.json` holds four NeMo Evaluator records of one GSM8K test question (question text and reference answer, with Qwen3-0.6B responses); `tests/fixtures/r11/longrun/*.items.jsonl.gz` hold per-item scores keyed by GSM8K test indices (no text).
- **Source:** GSM8K, <https://github.com/openai/grade-school-math> at commit `3101c7d5072418e28b9008a6636bde82a006892c` (Hugging Face `openai/gsm8k`, config `main`, revision `740312add88f781978c0658806c59bc2815b9866`).
- **License:** MIT. The license text, fetched from the repository at that commit:

```
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Qwen3-0.6B outputs

- **What:** the model responses in the same `raw_samples.json` were generated by Qwen3-0.6B (<https://huggingface.co/Qwen/Qwen3-0.6B>, Apache-2.0) and a GRPO-trained checkpoint of it.
