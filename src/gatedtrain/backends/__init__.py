"""Tier B backends (GPU, run by an operator; MVP-SPEC section 4).

Code only: the test suite never runs them. `nemo_rl` builds the launch command for
one NeMo RL recipe at 4f39fd8; `nemo_evaluator` reads NeMo Evaluator
`results.jsonl` at ea35ba3. Outputs are tagged `unpromotable` (01 section 8.1).
"""
