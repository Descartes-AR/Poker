"""Run the v2 simulation. Three opponents, fresh agent each, n_hands hands."""

import pandas as pd

from keyspace import PokerKeyspace
from simulation import PokerSimulation
from knowledge import initialize_all_knowledge
from opponents import calling_station, rock, loose_aggressive


def run_condition(opp_name, opp_policy, n_hands=5000, seed=4):
    root = PokerKeyspace()
    hand_log = []
    sim = PokerSimulation(f"sim_{opp_name}", root,
        opponent_policy=opp_policy,
        lr=1e-2, gamma=0.9, sd=0.5, eta_mcs=1e-3,
        hand_log=hand_log, seed=seed)

    initialize_all_knowledge(sim, sd=0.1)
    sim.kickoff_first_hand()

    for event in sim.run():
        if len(hand_log) >= n_hands: break
        # Watch for bust phase — terminate if either player busted
        d = root.d
        phase = sim.state.phase[0]
        if phase[~d.phase.bust] > 0.5:
            print(f"  bust detected at hand {len(hand_log)}; stopping")
            break

    df = pd.DataFrame(hand_log)
    df["opponent"] = opp_name
    return df


if __name__ == "__main__":
    all_dfs = []
    for name, pol in [("calling_station", calling_station),
                       ("rock", rock),
                       ("loose_aggressive", loose_aggressive)]:
        print(f"running {name} ...")
        df = run_condition(name, pol, n_hands=5000)
        df.to_csv(f"results_{name}.csv", index=False)
        all_dfs.append(df)
        print(f"  ... done, {len(df)} hands logged")
        # Bluff rate among hands where the agent actually held J
        j_hands = df[df["p1_card"] == "j"]
        bluff_rate = j_hands["is_bluff"].mean() if len(j_hands) else float("nan")
        print(f"  bluff rate (P[raise | held J]): {bluff_rate:.3f}  "
              f"(over {len(j_hands)} J-hands)")
        print(f"  cumulative profit: ${df['cum_profit'].iloc[-1]:+.2f} "
              f"over {len(df)} hands  ({df['reward_p1'].mean():+.3f}/hand)")
    pd.concat(all_dfs, ignore_index=True).to_csv("results_all.csv", index=False)
