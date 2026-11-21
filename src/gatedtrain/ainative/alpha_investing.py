"""AN-15: Foster-Stine alpha-investing arithmetic (pure functions, plain floats, no rounding).

Three alpha quantities, kept distinct everywhere:

- `alpha`  = alpha_j, the test level the bid grants (W_{j-1} / 2); the gate tests at this level.
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
    """alpha_j: half of the current wealth."""
    return wealth / 2


def can_bid(wealth: float) -> bool:
    return bid(wealth) >= MIN_BID


def settle(wealth: float, alpha: float, passed: bool) -> dict:
    """Settle one attempt that was tested at `alpha` (the level the bid granted)."""
    cost = 0.0 if passed else alpha / (1 - alpha)
    earned = PAYOUT if passed else 0.0
    return {"wealth_before": wealth, "alpha": alpha, "passed": bool(passed), "cost": cost, "earned": earned,
            "wealth_after": wealth - cost + earned}


def replay(settled: list[dict]) -> float:
    """Wealth after the settle rows, in order, from INITIAL_WEALTH; refuses a row whose `alpha` is not the bid of
    the replayed wealth (a tampered or reordered ledger)."""
    wealth = INITIAL_WEALTH
    for i, row in enumerate(settled):
        if not isinstance(row, dict) or row.get("alpha") != bid(wealth) \
                or not isinstance(row.get("passed"), bool):
            raise Refused([Problem("AN-15", f"ledger[{i}]", "the row is not the bid of the replayed wealth",
                                   "the ledger was altered; start a new test on new sealed data")])
        wealth = settle(wealth, row["alpha"], row["passed"])["wealth_after"]
    return wealth
