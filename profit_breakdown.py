"""
Profit decomposition — where does the money actually come from / go?

Tests the hypothesis: against the rock, the big losses are NOT from bluffing,
but from CHECKING or CALLING a raise while holding a Jack (instead of folding).
If true, bluffing could be net-positive while a separate leak (passive play with
J facing aggression) drowns it out.

It classifies every hand by the agent's LINE and reports, per opponent:
  - count of hands
  - total profit contributed by that line
  - mean profit per hand on that line
...so you can see exactly which lines win and which bleed money.

USAGE:
    python profit_breakdown.py              # 5000 hands per opponent
    python profit_breakdown.py 2000         # 2000 hands per opponent
"""

import sys
import pandas as pd

from run import run_condition
from opponents import calling_station, rock, loose_aggressive


def classify_line(row):
    """Label the agent's line from its card and action sequence.

    Raises are split by SIZE (min_raise = half pot, max_raise = full pot) so we
    can compare bet-sizing economics per card. The first raise action the agent
    took determines the raise-size label for the hand.
    """
    card = row["p1_card"]
    acts = row.get("p1_actions") or []

    first_raise = next((a for a in acts if a in ("min_raise", "max_raise")), None)
    called = "call" in acts
    folded = "fold" in acts
    checked = "check" in acts
    C = card.upper()

    if first_raise is not None:
        # Distinguish bluff-sizing for J vs value-sizing for Q/K.
        kind = "bluff" if card == "j" else "value"
        return f"{C}: {first_raise} ({kind})"
    if called:
        # For J, calling a raise is the 'passive vs raise' leak; keep that label.
        if card == "j":
            return "J: called (passive vs raise)"
        return f"{C}: called"
    if folded:
        return f"{C}: folded"
    if checked:
        # J checked with no raise faced is distinct (forfeits blind); others too.
        if card == "j":
            return "J: checked (no raise faced)"
        return f"{C}: checked"
    return f"{C}: other"


def analyze(opp_name, opp_policy, n_hands):
    df = run_condition(opp_name, opp_policy, n_hands=n_hands, seed=1)
    df["line"] = df.apply(classify_line, axis=1)

    total = df["reward_p1"].sum()
    print(f"\n{'='*72}\n{opp_name}  —  total profit ${total:+.0f} over {len(df)} hands\n{'='*72}")

    # Per-line breakdown, sorted by total contribution (most negative first)
    g = df.groupby("line")["reward_p1"].agg(["count", "sum", "mean"]).sort_values("sum")
    print(f"  {'line':32} {'hands':>6} {'tot $':>10} {'$/hand':>9}  {'%ofloss':>8}")
    total_loss = -df[df["reward_p1"] < 0]["reward_p1"].sum()
    for line, r in g.iterrows():
        share = (-r["sum"] / total_loss * 100) if (r["sum"] < 0 and total_loss > 0) else 0.0
        print(f"  {line:32} {int(r['count']):>6} {r['sum']:>+10.0f} {r['mean']:>+9.2f}  "
              f"{share:>7.1f}%")

    # Direct test: compare bluff SIZES and passive play with J
    jmin = df[df["line"] == "J: min_raise (bluff)"]["reward_p1"]
    jmax = df[df["line"] == "J: max_raise (bluff)"]["reward_p1"]
    jp = df[df["line"] == "J: called (passive vs raise)"]["reward_p1"]
    jf = df[df["line"] == "J: folded"]["reward_p1"]
    print(f"\n  BLUFF-SIZING TEST (Jack lines):")
    print(f"    J min_raise (bluff)   : {len(jmin):>5} hands, total ${jmin.sum():+.0f}, mean ${jmin.mean() if len(jmin) else 0:+.2f}")
    print(f"    J max_raise (bluff)   : {len(jmax):>5} hands, total ${jmax.sum():+.0f}, mean ${jmax.mean() if len(jmax) else 0:+.2f}")
    print(f"    J called vs raise     : {len(jp):>5} hands, total ${jp.sum():+.0f}, mean ${jp.mean() if len(jp) else 0:+.2f}")
    print(f"    J folded              : {len(jf):>5} hands, total ${jf.sum():+.0f}, mean ${jf.mean() if len(jf) else 0:+.2f}")
    if len(jmin) and len(jmax):
        better = "max_raise" if jmax.mean() > jmin.mean() else "min_raise"
        print(f"    -> Which bluff size is better? {better} "
              f"(min {jmin.mean():+.2f} vs max {jmax.mean():+.2f} per hand)")

    # Value-sizing for K (should be profitable; compare sizes)
    kmin = df[df["line"] == "K: min_raise (value)"]["reward_p1"]
    kmax = df[df["line"] == "K: max_raise (value)"]["reward_p1"]
    print(f"\n  VALUE-SIZING TEST (King lines):")
    print(f"    K min_raise (value)   : {len(kmin):>5} hands, total ${kmin.sum():+.0f}, mean ${kmin.mean() if len(kmin) else 0:+.2f}")
    print(f"    K max_raise (value)   : {len(kmax):>5} hands, total ${kmax.sum():+.0f}, mean ${kmax.mean() if len(kmax) else 0:+.2f}")
    if len(kmin) and len(kmax):
        better = "max_raise" if kmax.mean() > kmin.mean() else "min_raise"
        print(f"    -> Which value size is better? {better} "
              f"(min {kmin.mean():+.2f} vs max {kmax.mean():+.2f} per hand)")
    return df


def main():
    n_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    for name, pol in [("calling_station", calling_station),
                      ("rock", rock),
                      ("loose_aggressive", loose_aggressive)]:
        analyze(name, pol, n_hands)

    print(f"\n{'='*72}")
    print("READING THIS:")
    print("  • Lines are now split by RAISE SIZE (min_raise = ½ pot,")
    print("    max_raise = full pot) so you can compare bet-sizing economics.")
    print("  • Compare 'J: min_raise (bluff)' vs 'J: max_raise (bluff)': if one")
    print("    size loses far more, the leak may be a bet-SIZING / pot-accounting")
    print("    problem (e.g. over-committing on a full-pot bluff that gets called)")
    print("    rather than bluffing per se.")
    print("  • Sanity-check the King value lines: max_raise should generally")
    print("    earn MORE than min_raise vs a caller; if not, suspect the pot/stack")
    print("    accounting for that bet size in game.py.")
    print("  • Many '-5.00 $/hand' lines are just the forfeited $5 blind on")
    print("    unwinnable weak hands, not decision errors.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
