"""
Assembly: cognition + game-side components, all in one pyClarion Agent.

KEY CHANGE: ActionSensor now performs ACTION MASKING. It reads the Q-values
the network produced (over all 5 actions), filters to legal-in-current-state
actions, and writes the highest-Q legal action to state.action_taken.

This means the bottom-level Q-network can develop preferences for illegal
actions and we just override them at execution time. Cleaner than masking
during selection, less interference with the learning pipeline.
"""

from datetime import timedelta
from typing import Callable

from pyClarion import Agent, Input, Event, Priority
from pyClarion.events import ForwardUpdate, Site
from pyClarion.components.base import Component

from keyspace import PokerKeyspace
from game import (GameState, Dealer, Opponent, GameTracker, LEGAL_ACTIONS,
                  _read_hist, _onehot_action_for_player, _onehot_player,
                  _onehot_phase)
from nacs import NACSModule
from acs import ACSModule
from ms import MSModule
from mcs import MCSModule


class ActionSensor(Component):
    """Plays the SELECTOR's own choice. The selector selects over bottom.main,
    which the ACS has already augmented with the rule recommendations
    (learned_Q + rule_bias). So the played action is the combined, overridable
    choice AND exactly the action the TD update learns from — played == learned
    by construction, with no overwrite and no attribution race. Only if the
    selector lands on an illegal action does this fall back to the best legal
    action and realign the record."""

    def __init__(self, name, root, acs_module, state):
        super().__init__(name)
        self.root = root
        self.acs = acs_module
        self.state = state

    def resolve(self, event):
        if event.source == self.acs.selector.select:
            self.system.schedule(self.report_action())

    def report_action(self, dt=timedelta(), priority=Priority.PROPAGATION):
        d = self.root.d
        b = self.root.b
        wm = b.main.wm
        action_names = ["check", "min_raise", "max_raise", "call", "fold"]

        hist_str = _read_hist(self.state.history[0], d)
        legal_strs = LEGAL_ACTIONS.get(hist_str, [])

        # The selector's chosen action (its one-hot main output).
        sel = self.acs.selector.main[0]
        picked = None
        for an in action_names:
            if sel.d.get(~wm * ~getattr(d.action, an), 0.0) > 0.5:
                picked = an
                break

        if picked in legal_strs:
            # Played == what the selector chose == what it learns. No overwrite.
            return Event(self.report_action, [
                ForwardUpdate(self.state.action_taken,
                    _onehot_action_for_player(d, "p1", picked), "write"),
                ForwardUpdate(self.state.last_actor,
                    _onehot_player(d, "p1"), "write"),
            ], dt, priority)

        # Safety net: selector picked an illegal action (or none). Fall back to
        # the best LEGAL action by the combined value in bottom.main, and realign
        # the selector's record so learning targets the played action. Rule_bias
        # biases the selector toward legal actions, so this should be uncommon.
        combined = self.acs.bottom.main[0]
        vals = {a: combined.d.get(~wm * ~getattr(d.action, a), 0.0) for a in legal_strs}
        chosen = max(legal_strs, key=lambda a: vals[a]) if legal_strs else "fold"
        realign = {~wm * ~getattr(d.action, an): (1.0 if an == chosen else 0.0)
                   for an in action_names}
        return Event(self.report_action, [
            ForwardUpdate(self.state.action_taken,
                _onehot_action_for_player(d, "p1", chosen), "write"),
            ForwardUpdate(self.state.last_actor,
                _onehot_player(d, "p1"), "write"),
            ForwardUpdate(self.acs.selector.actions, realign, "write"),
        ], dt, priority)


class PokerSimulation(Agent):
    def __init__(self, name, root, opponent_policy: Callable[[dict], str],
                 *, lr=1e-2, gamma=0.9, sd=0.5, eta_mcs=1e-3,
                 hand_log=None, seed=None):
        super().__init__(name, root)
        self.root = root
        b, d = root.b, root.d
        feat_d = (b.main.wm, d)
        self.hand_log = hand_log if hand_log is not None else []

        with self:
            # Cognition
            self.ipt  = Input(f"{name}.ipt", feat_d)
            self.nacs = NACSModule(f"{name}.nacs", root)
            self.acs  = ACSModule(f"{name}.acs", root, self.nacs,
                                   lr=lr, gamma=gamma, sd=sd)
            self.ms   = MSModule(f"{name}.ms", root,
                                  reward_target=self.acs.selector.reward)
            self.mcs  = MCSModule(f"{name}.mcs", root, self.acs, self.ms,
                                   eta=eta_mcs)
            # Wire perception into cognitive pipeline
            self.ipt >> self.nacs.bu
            self.ipt >> self.acs.bu
            self.ipt >> self.acs.bottom
            # NOTE: MS writes its computed reward DIRECTLY onto the selector's
            # reward site (see MSModule.compute_reward). The `>>` operator only
            # wires main→input, so it cannot be used for the reward channel.

            # Game-side
            self.state    = GameState(f"{name}.state", root)
            self.dealer   = Dealer(f"{name}.dealer", root, self.state,
                                    self.ipt, selector=self.acs.selector,
                                    seed=seed)
            self.opponent = Opponent(f"{name}.opponent", root, self.state,
                                      opponent_policy)
            self.tracker  = GameTracker(f"{name}.tracker", root, self.state,
                                         self.ipt, self.ms.chip_in,
                                         selector=self.acs.selector,
                                         log_callback=self._log_hand)
            self.action_sensor = ActionSensor(f"{name}.sensor", root,
                                               self.acs, self.state)

    def _log_hand(self, hand_data):
        actions = hand_data["actions"]
        p1_first = next(((p,a) for p,a in actions if p == "p1"), None)
        # Bluff definition for the new action space:
        # P1 raises (min or max) with J on their first action.
        is_bluff = (p1_first is not None
                    and p1_first[1] in ("min_raise", "max_raise")
                    and hand_data["p1_card"] == "j")
        # Cumulative profit (running sum of per-hand rewards) — meaningful even
        # in independent-hands mode where per-hand stacks reset.
        prev_cum = self.hand_log[-1]["cum_profit"] if self.hand_log else 0.0
        cum_profit = prev_cum + hand_data["reward_p1"]
        self.hand_log.append({
            "hand_idx": len(self.hand_log),
            "p1_card": hand_data["p1_card"],
            "p2_card": hand_data["p2_card"],
            "p1_first_action": p1_first[1] if p1_first else None,
            "p1_actions": [a for (p, a) in actions if p == "p1"],
            "all_actions": list(actions),
            "reward_p1": hand_data["reward_p1"],
            "cum_profit": cum_profit,
            "pot_won": hand_data["pot_won"],
            "winner": hand_data["winner"],
            "ended_by": hand_data["ended_by"],
            "is_bluff": is_bluff,
            "n_actions": len(actions),
            "agent_stack": self.state.agent_stack[0][~self.root.d.money.amount],
            "opp_stack": self.state.opp_stack[0][~self.root.d.money.amount],
        })

    def kickoff_first_hand(self):
        d = self.root.d
        self.system.schedule(Event(
            source=self.kickoff_first_hand,
            updates=[ForwardUpdate(self.state.phase,
                _onehot_phase(d, "hand_start"), "write")],
            time=timedelta(),
            priority=Priority.PROPAGATION,
        ))
