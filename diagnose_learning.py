"""
Three diagnostics to determine whether the bottom level is learning at all.

  CHECK 1 — Reward delivery: is chip_delta arriving at the MS with the right
            SIGN and MAGNITUDE? (Bluffing into a calling station and being
            called should produce clearly NEGATIVE chip_delta.)

  CHECK 2 — Weight drift: do the bottom-layer Q-network weights actually CHANGE
            between hand 0 and hand N? (If nearly identical, the optimizer
            is a no-op and the agent plays a fixed random policy forever.)

  CHECK 3 — Q-by-opponent: is Q(max_raise | J) DIFFERENT across opponents?
            (If identical, the bottom level is blind to the opponent and the
            flat bluff rate is explained.)

Run:
    python diagnose_learning.py
"""

import copy
import numpy as np

from keyspace import PokerKeyspace
from simulation import PokerSimulation
from knowledge import initialize_all_knowledge
from opponents import calling_station, rock, loose_aggressive


# ─── helpers ─────────────────────────────────────────────────────────────
def _weights_snapshot(sim):
    """Return a plain dict copy of the bottom layer's weights site."""
    return {k: v for k, v in sim.acs.bottom.weights[0].d.items()}


def _weights_l2_change(w0, w1):
    """L2 norm of the difference between two weight snapshots."""
    keys = set(w0) | set(w1)
    diffs = [w1.get(k, 0.0) - w0.get(k, 0.0) for k in keys]
    return float(np.sqrt(sum(x * x for x in diffs)))


def _q_for(sim, card_name, hist_name, action_name):
    """Read the bottom-level Q-value for (card, history) → action by pushing
    that observation through the bottom layer directly and reading its main.

    We do this by writing the one-hot feature input to the bottom layer's
    input site and triggering a forward pass in isolation.
    """
    d, b = sim.root.d, sim.root.b
    wm = b.main.wm
    # The bottom layer's current main (Q-values) reflects the LAST forward
    # pass. To probe a specific state we read the weights and compute the
    # linear output by hand: Q(a) = sum_i w[feat_i, a] * x[feat_i] + bias[a].
    weights = sim.acs.bottom.weights[0].d
    bias = sim.acs.bottom.bias[0].d
    card_key = ~wm * ~getattr(d.card, card_name)
    hist_key = ~wm * ~getattr(d.history, hist_name)
    action_key = ~wm * ~getattr(d.action, action_name)
    q = bias.get(action_key, 0.0)
    for feat_key in (card_key, hist_key):
        wkey = feat_key * action_key
        q += weights.get(wkey, 0.0)
    return q


def run_with_diagnostics(opp_name, opp_policy, n_hands=5000, seed=4):
    root = PokerKeyspace()
    hand_log = []
    sim = PokerSimulation(f"diag_{opp_name}", root,
        opponent_policy=opp_policy, hand_log=hand_log, seed=seed)
    initialize_all_knowledge(sim, sd=0.1)

    # ── CHECK 2 setup: snapshot weights BEFORE any learning ──
    w_before = _weights_snapshot(sim)

    # ── CHECK 1 setup: monkey-patch MS.compute_reward to record chip_delta ──
    chip_deltas = []
    orig_compute = sim.ms.compute_reward
    d, b = root.d, root.b
    def patched_compute(dt=None, priority=None, _orig=orig_compute):
        # Read chip_delta the same way compute_reward does, and record it
        cd = sim.ms.chip_in[0][~b.main.rwd * ~d.reward.main]
        chip_deltas.append(cd)
        # call original with its defaults
        if dt is None:
            return _orig()
        return _orig(dt, priority)
    sim.ms.compute_reward = patched_compute

    # Run
    sim.kickoff_first_hand()
    for _ in sim.run():
        if len(hand_log) >= n_hands:
            break

    # ── CHECK 2: weights after ──
    w_after = _weights_snapshot(sim)
    l2 = _weights_l2_change(w_before, w_after)

    # ── CHECK 3: Q(max_raise | J, start) and Q(fold | J, p1_max_raise) ──
    q_jraise = _q_for(sim, "j", "start", "max_raise")
    q_jcheck = _q_for(sim, "j", "start", "check")

    # ── CHECK 1 analysis: chip_delta on bluff hands vs all ──
    import pandas as pd
    df = pd.DataFrame(hand_log)
    bluff_hands = df[(df.p1_card == "j") & (df.is_bluff)]
    mean_bluff_reward = bluff_hands["reward_p1"].mean() if len(bluff_hands) else float("nan")

    return {
        "opp": opp_name,
        "n_hands": len(hand_log),
        "chip_deltas_recorded": len(chip_deltas),
        "chip_delta_mean": float(np.mean(chip_deltas)) if chip_deltas else float("nan"),
        "chip_delta_min": float(np.min(chip_deltas)) if chip_deltas else float("nan"),
        "chip_delta_max": float(np.max(chip_deltas)) if chip_deltas else float("nan"),
        "chip_delta_nonzero_frac": float(np.mean([abs(x) > 1e-9 for x in chip_deltas])) if chip_deltas else float("nan"),
        "weight_l2_change": l2,
        "n_weights": len(w_after),
        "q_jraise_start": q_jraise,
        "q_jcheck_start": q_jcheck,
        "bluff_rate_J": float(df[df.p1_card == "j"]["is_bluff"].mean()),
        "mean_reward_on_bluff": mean_bluff_reward,
        "cum_profit": float(df["cum_profit"].iloc[-1]),
    }


def main():
    print("=" * 72)
    print("LEARNING DIAGNOSTICS — short runs (5000 hands each)")
    print("=" * 72)

    results = []
    for name, pol in [("calling_station", calling_station),
                       ("rock", rock),
                       ("loose_aggressive", loose_aggressive)]:
        print(f"\nrunning {name} ...")
        r = run_with_diagnostics(name, pol, n_hands=5000, seed=4)
        results.append(r)

        print(f"\n  ── CHECK 1: REWARD DELIVERY ({name}) ──")
        print(f"     chip_deltas recorded:     {r['chip_deltas_recorded']}  (should ≈ n_hands={r['n_hands']})")
        print(f"     chip_delta mean:          {r['chip_delta_mean']:+.3f}")
        print(f"     chip_delta min / max:     {r['chip_delta_min']:+.2f} / {r['chip_delta_max']:+.2f}")
        print(f"     fraction nonzero:         {r['chip_delta_nonzero_frac']:.3f}  (0.0 = reward never arrives!)")
        print(f"     mean reward on J-bluffs:  {r['mean_reward_on_bluff']:+.3f}")
        if name == "calling_station":
            print(f"        ↑ EXPECT NEGATIVE: bluffing a calling station loses. If ≥0, reward sign is wrong.")

        print(f"\n  ── CHECK 2: WEIGHT DRIFT ({name}) ──")
        print(f"     bottom-layer weights:     {r['n_weights']} entries")
        print(f"     L2 change over run:       {r['weight_l2_change']:.6f}")
        print(f"        ↑ NEAR ZERO (<1e-4) = optimizer is a no-op, NO learning happening.")

        print(f"\n  ── CHECK 3: Q-VALUES ({name}) ──")
        print(f"     Q(max_raise | J,start):   {r['q_jraise_start']:+.4f}")
        print(f"     Q(check     | J,start):   {r['q_jcheck_start']:+.4f}")
        print(f"     bluff rate (J):           {r['bluff_rate_J']:.3f}")
        print(f"     cumulative profit:        ${r['cum_profit']:+.2f}")

    # Cross-opponent comparison for CHECK 3
    print("\n" + "=" * 72)
    print("CROSS-OPPONENT COMPARISON (the decisive test)")
    print("=" * 72)
    print(f"{'opponent':18s} {'Q(raise|J)':>12s} {'Q(check|J)':>12s} {'bluff_rate':>11s} {'wt_change':>11s}")
    for r in results:
        print(f"{r['opp']:18s} {r['q_jraise_start']:>+12.4f} {r['q_jcheck_start']:>+12.4f} "
              f"{r['bluff_rate_J']:>11.3f} {r['weight_l2_change']:>11.5f}")
    print()
    print("INTERPRETATION:")
    print("  • If Q(raise|J) is ~IDENTICAL across opponents → bottom level is blind to")
    print("    opponent → it isn't learning opponent-specific value. THE core problem.")
    print("  • If weight L2 change is ~0 everywhere → optimizer never updates weights.")
    print("  • If chip_delta is always 0 or wrong sign → reward channel is broken.")
    print("  • If weights DO change and Q DIFFERS but bluff rate is still flat → the")
    print("    Pool/selector isn't translating learned Q into behavior (cam or temp issue).")


if __name__ == "__main__":
    main()
