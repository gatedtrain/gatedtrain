"""AN-15: Foster-Stine alpha-investing arithmetic (pure functions, plain floats, no rounding).

Three alpha quantities, kept distinct everywhere:

- `alpha`  = alpha_j, the test level the bid grants (W_{j-1} / 2, capped at W_{j-1} / (1 + W_{j-1}) so that the cost
  never exceeds the wealth; the cap binds only above W = 1); the gate tests at this level.
- `cost`   = the wealth charged: alpha_j / (1 - alpha_j) when the attempt did not pass, 0.0 on a PASS.
- `earned` = the payout: PAYOUT on a PASS, 0.0 otherwise.

Wealth starts at INITIAL_WEALTH; an attempt is refused when its bid is below MIN_BID.
"""

from __future__ import annotations

from ..errors import Problem, Refused

TARGET_ALPHA = 0.05
INITIAL_WEALTH = 0.05 * (1 - 0.05)
PAYOUT = 0.05
MIN_BID = 0.005
PLANNING_ALPHA = INITIAL_WEALTH / 2
RULE = "foster-stine"
GUARANTEE = ("alpha-investing (Foster-Stine) over the gate attempts on this sealed set controls the marginal false "
             "discovery rate (mFDR) at level 0.05")


def bid(wealth: float) -> float:
    """alpha_j: half of the current wealth, and at most W / (1 + W), whose cost alpha_j / (1 - alpha_j) is W itself, so
    a bid never costs more than the wealth. The two are equal at W = 1, so below it the bid is W / 2 exactly."""
    return min(wealth / 2, wealth / (1 + wealth))


def can_bid(wealth: float) -> bool:
    return bid(wealth) >= MIN_BID


def settle(wealth: float, alpha: float, passed: bool) -> dict:
    """Settle one attempt that was tested at `alpha` (the level the bid granted)."""
    cost = 0.0 if passed else alpha / (1 - alpha)
    earned = PAYOUT if passed else 0.0
    return {"wealth_before": wealth, "alpha": alpha, "passed": bool(passed), "cost": cost, "earned": earned,
            "wealth_after": wealth - cost + earned}


_STORED = ("wealth_before", "cost", "earned", "wealth_after")


def replay(settled: list[dict]) -> float:
    """Wealth after the settle rows, in order, from INITIAL_WEALTH. Refuses (AN-15) a row granted when no bid was
    allowed (`can_bid` false: a ledger past its run-out), a row whose `alpha` is not the bid of the replayed wealth, and
    a row whose stored `wealth_before`, `cost`, `earned` or `wealth_after` (each one the row holds) differs from the
    recomputed value: a tampered, reordered or truncated ledger."""
    wealth = INITIAL_WEALTH
    for i, row in enumerate(settled):
        if not isinstance(row, dict) or not can_bid(wealth) or row.get("alpha") != bid(wealth) \
                or not isinstance(row.get("passed"), bool):
            raise Refused([Problem("AN-15", f"ledger[{i}]", "the row is not the bid of the replayed wealth",
                                   "the ledger was altered; start a new test on new sealed data")])
        again = settle(wealth, row["alpha"], row["passed"])
        if any(key in row and row[key] != again[key] for key in _STORED):
            raise Refused([Problem("AN-15", f"ledger[{i}]", "the row's wealth, cost or payout differs from the "
                                   "replayed arithmetic",
                                   "the ledger was altered; start a new test on new sealed data")])
        wealth = again["wealth_after"]
    return wealth
