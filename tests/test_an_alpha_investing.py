"""AN-15: Foster-Stine alpha-investing arithmetic (tests/README.md section 13).

W0 = 0.05 * (1 - 0.05); bid alpha_j = W_{j-1} / 2; a non-PASS costs alpha_j / (1 - alpha_j); a PASS earns 0.05;
an attempt is refused when alpha_j < 0.005; `replay` refuses a row whose alpha is not the bid of the replayed wealth.
"""

from __future__ import annotations

import unittest

from gatedtrain.ainative import alpha_investing as ai
from gatedtrain.ainative.trust import FixedAlphaLedger, InProcessTrust
from gatedtrain.errors import Refused


class AlphaInvestingTest(unittest.TestCase):
    def test_constants(self) -> None:
        self.assertEqual(ai.TARGET_ALPHA, 0.05)
        self.assertEqual(ai.INITIAL_WEALTH, 0.0475)
        self.assertEqual(ai.PAYOUT, 0.05)
        self.assertEqual(ai.MIN_BID, 0.005)
        self.assertEqual(ai.PLANNING_ALPHA, 0.02375)
        self.assertEqual(ai.RULE, "foster-stine")
        self.assertIn("marginal false discovery rate (mFDR) at level 0.05", ai.GUARANTEE)

    def test_three_failures_then_refusal(self) -> None:
        w, bids = ai.INITIAL_WEALTH, []
        while ai.can_bid(w):
            a = ai.bid(w)
            bids.append(a)
            w = ai.settle(w, a, False)["wealth_after"]
        for got, want in zip(bids, [0.02375, 0.011586107554417413, 0.005725148070253095]):
            self.assertAlmostEqual(got, want, places=15)
        self.assertEqual(len(bids), 3)
        self.assertAlmostEqual(ai.bid(w), 0.0028460910071375026, places=15)
        self.assertFalse(ai.can_bid(w))

    def test_pass_earns_payout_without_cost(self) -> None:
        row = ai.settle(0.0475, 0.02375, True)
        self.assertEqual((row["cost"], row["earned"]), (0.0, 0.05))
        self.assertAlmostEqual(row["wealth_after"], 0.0975)
        self.assertEqual(set(row), {"wealth_before", "alpha", "passed", "cost", "earned", "wealth_after"})
        self.assertEqual((row["wealth_before"], row["alpha"], row["passed"]), (0.0475, 0.02375, True))

    def test_bid_never_costs_more_than_the_wealth(self) -> None:     # final review 6d
        self.assertEqual(ai.bid(3.0), 0.75)                            # W / (1 + W) above W = 1, not W / 2
        row = ai.settle(3.0, ai.bid(3.0), False)
        self.assertEqual(row["cost"], 3.0)
        self.assertEqual(row["wealth_after"], 0.0)
        for w in (0.0475, 0.0975, 0.5, 1.0):                           # unchanged up to W = 1
            self.assertEqual(ai.bid(w), w / 2)

    def test_fail_cost(self) -> None:
        row = ai.settle(0.0475, 0.02375, False)
        self.assertAlmostEqual(row["cost"], 0.024327784891165175)
        self.assertEqual(row["earned"], 0.0)
        self.assertAlmostEqual(row["wealth_after"], 0.0475 - 0.024327784891165175)

    def test_replay_from_start_and_over_rows(self) -> None:
        self.assertEqual(ai.replay([]), ai.INITIAL_WEALTH)
        rows, w = [], ai.INITIAL_WEALTH
        for passed in (False, True, False):
            row = ai.settle(w, ai.bid(w), passed)
            rows.append(row)
            w = row["wealth_after"]
        self.assertEqual(ai.replay(rows), w)

    def test_replay_refuses_a_wrong_bid(self) -> None:
        good = ai.settle(ai.INITIAL_WEALTH, ai.bid(ai.INITIAL_WEALTH), False)
        bad = ai.settle(good["wealth_after"], 0.02, False)
        with self.assertRaises(Refused) as cm:
            ai.replay([good, bad])
        self.assertEqual(cm.exception.problems[0].rule, "AN-15")
        bad_first = dict(good, alpha=0.05)
        with self.assertRaises(Refused):
            ai.replay([bad_first])

    def test_fixed_ledger_planning_alpha(self) -> None:
        self.assertEqual(FixedAlphaLedger(None).planning_alpha, 0.05)    # type: ignore[arg-type]
        self.assertEqual(FixedAlphaLedger(None, alpha=0.2).planning_alpha, 0.2)  # type: ignore[arg-type]

    def test_in_process_trust_mode(self) -> None:
        self.assertEqual(InProcessTrust.mode, "in_process")


if __name__ == "__main__":
    unittest.main()
