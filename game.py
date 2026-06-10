"""
Game-side Components — variable bet sizing with pot/stack tracking.

GameState now carries pot, agent_stack, opp_stack on Sites.
Dealer posts blinds (subtract $5 from each stack, pot = $10).
GameTracker computes raise amounts as fractions of current pot,
moves chips between stacks and pot, transfers pot to winner on terminal.

Opponent policy signature changed to receive pot/stacks in addition to
card/history, so opponents can play stack-aware strategies if desired.
"""

import random
from datetime import timedelta
from typing import Callable

from pyClarion import Priority, Event, Input
from pyClarion.events import ForwardUpdate, State, Site
from pyClarion.components.base import Component

from keyspace import PokerKeyspace


# ─── Constants ───────────────────────────────────────────────────────────
STARTING_STACK = 120.0
BLIND = 6.0
INITIAL_POT = 2 * BLIND   # $12  (even, so half/full-pot raises and splits stay even)
_RANK = {"j": 0, "q": 1, "k": 2}


# Legal actions per history (action availability mask)
LEGAL_ACTIONS = {
    "start":           ["check", "min_raise", "max_raise"],
    "p1_checked":      ["check", "min_raise", "max_raise"],
    "p1_min_raise":    ["call", "fold"],
    "p1_max_raise":    ["call", "fold"],
    "check_min_raise": ["call", "fold"],
    "check_max_raise": ["call", "fold"],
}


def _card_atom(d, name): return {"j": d.card.j, "q": d.card.q, "k": d.card.k}[name]


def _hist_atom(d, name):
    return getattr(d.history, name)


# ─── One-hot write helpers ───────────────────────────────────────────────
# pyClarion's "write" update MERGES into a site rather than replacing it.
# For categorical (one-hot) sites we must therefore explicitly zero every
# other atom in the sort, or stale values accumulate (e.g. a phase site
# ending up with both hand_start=1 and agent_turn=1, causing re-trigger loops).

_PHASE_NAMES = ["hand_start", "agent_turn", "opponent_turn", "terminal", "bust"]
_HIST_NAMES = ["start", "p1_checked", "p1_min_raise", "p1_max_raise",
               "check_min_raise", "check_max_raise",
               "terminal_showdown", "terminal_fold"]
_CARD_NAMES = ["j", "q", "k"]
_PLAYER_NAMES = ["p1", "p2"]
_ACTION_NAMES = ["check", "min_raise", "max_raise", "call", "fold"]


def _onehot_phase(d, active: str) -> dict:
    return {~getattr(d.phase, n): (1.0 if n == active else 0.0)
            for n in _PHASE_NAMES}


def _onehot_hist(d, active: str) -> dict:
    return {~getattr(d.history, n): (1.0 if n == active else 0.0)
            for n in _HIST_NAMES}


def _onehot_card(d, active: str) -> dict:
    return {~getattr(d.card, n): (1.0 if n == active else 0.0)
            for n in _CARD_NAMES}


def _onehot_player(d, active: str) -> dict:
    return {~getattr(d.player, n): (1.0 if n == active else 0.0)
            for n in _PLAYER_NAMES}


def _onehot_action_for_player(d, player: str, action: str) -> dict:
    """Full (player × action) one-hot — only the (player, action) cell is 1."""
    out = {}
    for pl in _PLAYER_NAMES:
        for ac in _ACTION_NAMES:
            key = ~getattr(d.player, pl) * ~getattr(d.action, ac)
            out[key] = 1.0 if (pl == player and ac == action) else 0.0
    return out


# ─── GameState ───────────────────────────────────────────────────────────
class GameState(Component):
    """Holds the world. New money sites: pot, agent_stack, opp_stack."""
    card_p1: Site = Site()
    card_p2: Site = Site()
    history: Site = Site()
    phase: Site = Site()
    action_taken: Site = Site()
    last_actor: Site = Site()
    pot: Site = Site()
    agent_stack: Site = Site()
    opp_stack: Site = Site()
    last_raise: Site = Site()   # scalar; the most recent raise amount (for call/fold accounting)

    def __init__(self, name, root: PokerKeyspace) -> None:
        super().__init__(name)
        self.root = root
        d = root.d
        (idx_card, idx_hist, idx_phase, idx_player,
         idx_action, idx_money) = self._init_indexes(
            d.card, d.history, d.phase, d.player, d.action, d.money)
        self.card_p1 = State(idx_card, {}, 0.0)
        self.card_p2 = State(idx_card, {}, 0.0)
        self.history = State(idx_hist, {}, 0.0)
        self.phase = State(idx_phase, {}, 0.0)
        self.action_taken = State(idx_player * idx_action, {}, 0.0)
        self.last_actor = State(idx_player, {}, 0.0)
        # Money sites — initial values set by Dealer
        self.pot = State(idx_money, {~d.money.amount: 0.0}, 0.0)
        self.agent_stack = State(idx_money,
            {~d.money.amount: STARTING_STACK}, 0.0)
        self.opp_stack = State(idx_money,
            {~d.money.amount: STARTING_STACK}, 0.0)
        self.last_raise = State(idx_money, {~d.money.amount: 0.0}, 0.0)


# ─── Dealer ──────────────────────────────────────────────────────────────
class Dealer(Component):
    """On phase=hand_start, posts blinds, deals cards, pushes initial obs.

    Also checks bust condition: if either stack < BLIND, sets phase=bust.
    """
    def __init__(self, name, root, state, agent_ipt, *, selector=None,
                 seed=None, independent_hands=True):
        super().__init__(name)
        self.root = root
        self.state = state
        self.agent_ipt = agent_ipt
        self.selector = selector       # ACS TDLearning; triggered after dealing
        self._rng = random.Random(seed)
        # If True, both stacks reset to STARTING_STACK every hand (each hand is
        # an independent training episode; nobody ever busts). If False, stacks
        # carry over and a player can bust — useful for a bankroll experiment,
        # but it halts learning prematurely, so it's off by default.
        self.independent_hands = independent_hands

    def resolve(self, event):
        forward = event.index(ForwardUpdate)
        if self.state.phase in forward:
            d = self.root.d
            if self.state.phase[0][~d.phase.hand_start] > 0.5:
                self.system.schedule(self.deal())

    def deal(self, dt=timedelta(), priority=Priority.PROPAGATION) -> Event:
        d = self.root.d
        b = self.root.b

        if self.independent_hands:
            # Each hand is an independent episode: reset both stacks to full,
            # then post blinds. Nobody ever busts; learning continues forever.
            agent_money = STARTING_STACK
            opp_money = STARTING_STACK
        else:
            # Carryover mode: read current stacks, check bust.
            agent_money = self.state.agent_stack[0][~d.money.amount]
            opp_money = self.state.opp_stack[0][~d.money.amount]
            if agent_money < BLIND or opp_money < BLIND:
                return Event(self.deal,
                    [ForwardUpdate(self.state.phase, _onehot_phase(d, "bust"), "write")],
                    dt, priority)

        # Post blinds
        new_agent_stack = agent_money - BLIND
        new_opp_stack   = opp_money - BLIND
        new_pot         = INITIAL_POT

        # Deal cards from a 12-card deck (4 each of J, Q, K), without
        # replacement. Unlike the old 1-each deck, the two dealt cards can now
        # share a rank (e.g. both Q) -> a tie at showdown, which splits the pot.
        deck = ["j", "q", "k"] * 4
        self._rng.shuffle(deck)
        c1, c2 = deck[0], deck[1]

        updates = [
            ForwardUpdate(self.state.card_p1,     _onehot_card(d, c1), "write"),
            ForwardUpdate(self.state.card_p2,     _onehot_card(d, c2), "write"),
            ForwardUpdate(self.state.history,     _onehot_hist(d, "start"), "write"),
            ForwardUpdate(self.state.phase,       _onehot_phase(d, "agent_turn"), "write"),
            ForwardUpdate(self.state.pot,         {~d.money.amount: new_pot}, "write"),
            ForwardUpdate(self.state.agent_stack, {~d.money.amount: new_agent_stack}, "write"),
            ForwardUpdate(self.state.opp_stack,   {~d.money.amount: new_opp_stack}, "write"),
            ForwardUpdate(self.state.last_raise,  {~d.money.amount: 0.0}, "write"),
            # Push agent perception: card + history.start
            ForwardUpdate(self.agent_ipt.main, {
                ~b.main.wm * ~_card_atom(d, c1): 1.0,
                ~b.main.wm * ~d.history.start: 1.0,
            }, "push"),
        ]
        # After dealing + pushing perception, trigger the agent's decision.
        # Scheduled as a slightly-later event so the perception write lands first.
        if self.selector is not None:
            self.system.schedule(self.selector.trigger(
                dt=timedelta(microseconds=1)))
        return Event(self.deal, updates, dt, priority)


# ─── Opponent ────────────────────────────────────────────────────────────
class Opponent(Component):
    """Watches phase. On opponent_turn, reads game state, computes action
    via self.policy, writes to action_taken."""

    def __init__(self, name, root, state,
                 policy: Callable[[dict], str]) -> None:
        """policy receives a dict with keys: card, history, pot,
        own_stack, opp_stack, last_raise → returns one of LEGAL_ACTIONS[history]."""
        super().__init__(name)
        self.root = root
        self.state = state
        self.policy = policy

    def resolve(self, event):
        forward = event.index(ForwardUpdate)
        if self.state.phase in forward:
            d = self.root.d
            if self.state.phase[0][~d.phase.opponent_turn] > 0.5:
                self.system.schedule(self.act())

    def act(self, dt=timedelta(), priority=Priority.PROPAGATION) -> Event:
        d = self.root.d
        ctx = {
            "card": _read_oh(self.state.card_p2[0], d.card, ["j","q","k"]),
            "history": _read_hist(self.state.history[0], d),
            "pot": self.state.pot[0][~d.money.amount],
            "own_stack": self.state.opp_stack[0][~d.money.amount],
            "opp_stack": self.state.agent_stack[0][~d.money.amount],
            "last_raise": self.state.last_raise[0][~d.money.amount],
        }
        action_str = self.policy(ctx)
        # Validate the policy chose a legal action
        legal = LEGAL_ACTIONS.get(ctx["history"], [])
        if action_str not in legal:
            # Fallback: pick first legal action
            action_str = legal[0]
        return Event(self.act, [
            ForwardUpdate(self.state.action_taken,
                _onehot_action_for_player(d, "p2", action_str), "write"),
            ForwardUpdate(self.state.last_actor,
                _onehot_player(d, "p2"), "write"),
        ], dt, priority)


# ─── GameTracker (the heart of pot/stack accounting) ─────────────────────
class GameTracker(Component):
    """On action_taken update, advances the game with proper pot/stack
    accounting. Handles raise sizing, calls, folds, terminal showdown."""

    def __init__(self, name, root, state, agent_ipt, ms_chip_in,
                 *, selector=None, log_callback=None) -> None:
        super().__init__(name)
        self.root = root
        self.state = state
        self.agent_ipt = agent_ipt
        self.ms_chip_in = ms_chip_in
        self.selector = selector       # triggered when turn handed back to agent
        self.log_callback = log_callback
        # Internal: track action history for terminal detection
        self.actions_seq: list[tuple[str, str]] = []
        # Track agent's stack BEFORE this hand began (for reward computation)
        self._stack_at_hand_start: float | None = None

    def resolve(self, event):
        forward = event.index(ForwardUpdate)
        # Snapshot agent's stack right after blinds are posted (i.e., when
        # phase becomes agent_turn for the first time of the hand)
        if self.state.phase in forward:
            d = self.root.d
            if self.state.phase[0][~d.phase.agent_turn] > 0.5 and not self.actions_seq:
                self._stack_at_hand_start = (
                    self.state.agent_stack[0][~d.money.amount] + BLIND
                )  # +BLIND because the blind was just posted
        if self.state.action_taken in forward:
            self.system.schedule(self.advance())

    def advance(self, dt=timedelta(), priority=Priority.PROPAGATION) -> Event:
        d = self.root.d
        b = self.root.b
        actor = _read_oh(self.state.last_actor[0], d.player, ["p1","p2"])
        action = _read_action(self.state.action_taken[0], d, actor)
        self.actions_seq.append((actor, action))

        seq = [a for _, a in self.actions_seq]
        pot = self.state.pot[0][~d.money.amount]
        last_raise = self.state.last_raise[0][~d.money.amount]
        agent_stack = self.state.agent_stack[0][~d.money.amount]
        opp_stack = self.state.opp_stack[0][~d.money.amount]

        # Decide raise amount if it's a raise action
        if action == "min_raise":   raise_amount = pot * 0.5
        elif action == "max_raise": raise_amount = pot * 1.0
        else:                       raise_amount = 0.0

        # Apply the action's effect on pot/stacks
        updates = []
        terminal = False
        next_hist = None
        winner_p1 = None     # None if showdown needed; bool at a decided showdown
        is_tie = False       # True at a showdown where both cards share a rank
        pot_after_action = pot

        if action == "check":
            # No money moves
            next_hist = self._after_check(seq, actor)
            if next_hist == "terminal_showdown":
                terminal = True
                r1 = _RANK[_read_oh(self.state.card_p1[0], d.card, ["j","q","k"])]
                r2 = _RANK[_read_oh(self.state.card_p2[0], d.card, ["j","q","k"])]
                is_tie = (r1 == r2)
                winner_p1 = (r1 > r2)
        elif action in ("min_raise", "max_raise"):
            # Actor pays raise_amount into the pot
            if actor == "p1":
                new_stack = agent_stack - raise_amount
                updates.append(ForwardUpdate(self.state.agent_stack,
                    {~d.money.amount: new_stack}, "write"))
            else:
                new_stack = opp_stack - raise_amount
                updates.append(ForwardUpdate(self.state.opp_stack,
                    {~d.money.amount: new_stack}, "write"))
            pot_after_action = pot + raise_amount
            updates.append(ForwardUpdate(self.state.pot,
                {~d.money.amount: pot_after_action}, "write"))
            updates.append(ForwardUpdate(self.state.last_raise,
                {~d.money.amount: raise_amount}, "write"))
            next_hist = self._after_raise(seq, actor, action)
        elif action == "call":
            # Actor pays last_raise into the pot
            call_amount = last_raise
            if actor == "p1":
                new_stack = agent_stack - call_amount
                updates.append(ForwardUpdate(self.state.agent_stack,
                    {~d.money.amount: new_stack}, "write"))
            else:
                new_stack = opp_stack - call_amount
                updates.append(ForwardUpdate(self.state.opp_stack,
                    {~d.money.amount: new_stack}, "write"))
            pot_after_action = pot + call_amount
            updates.append(ForwardUpdate(self.state.pot,
                {~d.money.amount: pot_after_action}, "write"))
            # Call always ends the hand (no re-raises in v2)
            terminal = True
            next_hist = "terminal_showdown"
            r1 = _RANK[_read_oh(self.state.card_p1[0], d.card, ["j","q","k"])]
            r2 = _RANK[_read_oh(self.state.card_p2[0], d.card, ["j","q","k"])]
            is_tie = (r1 == r2)
            winner_p1 = (r1 > r2)
        elif action == "fold":
            # Folder loses; opponent wins pot
            terminal = True
            next_hist = "terminal_fold"
            winner_p1 = (actor == "p2")    # P2 folded → P1 wins

        # On terminal, transfer pot (or split it on a tie)
        if terminal:
            if is_tie:
                # SPLIT POT. At any showdown both players have matched their
                # contributions, so returning half the (even) pot to each gives
                # each player back exactly what they put in -> chip_delta = 0 for
                # both. A true neutral outcome: not a win, not a loss, so it
                # cannot bias the agent's greedy exploration in either direction.
                half = pot_after_action / 2.0
                # Apply this action's not-yet-applied commitment, then refund half.
                p1_pending = (raise_amount if (action in ("min_raise","max_raise") and actor == "p1")
                              else last_raise if (action == "call" and actor == "p1")
                              else 0.0)
                p2_pending = (raise_amount if (action in ("min_raise","max_raise") and actor == "p2")
                              else last_raise if (action == "call" and actor == "p2")
                              else 0.0)
                new_agent_stack = agent_stack - p1_pending + half
                new_opp_stack   = opp_stack   - p2_pending + half
                updates.append(ForwardUpdate(self.state.agent_stack,
                    {~d.money.amount: new_agent_stack}, "write"))
                updates.append(ForwardUpdate(self.state.opp_stack,
                    {~d.money.amount: new_opp_stack}, "write"))
                updates.append(ForwardUpdate(self.state.pot,
                    {~d.money.amount: 0.0}, "write"))
                updates.append(ForwardUpdate(self.state.history,
                    _onehot_hist(d, next_hist), "write"))
                updates.append(ForwardUpdate(self.state.phase,
                    _onehot_phase(d, "terminal"), "write"))
                final_agent_stack = new_agent_stack
            else:
                winner_stack_site = self.state.agent_stack if winner_p1 \
                                    else self.state.opp_stack
                cur_winner_stack = (agent_stack if winner_p1 else opp_stack)
                # Account for stack changes from THIS action that haven't yet been applied
                if action in ("min_raise", "max_raise") and ((actor == "p1") == winner_p1):
                    cur_winner_stack -= raise_amount
                elif action == "call" and ((actor == "p1") == winner_p1):
                    cur_winner_stack -= last_raise
                new_winner_stack = cur_winner_stack + pot_after_action
                updates.append(ForwardUpdate(winner_stack_site,
                    {~d.money.amount: new_winner_stack}, "write"))
                updates.append(ForwardUpdate(self.state.pot,
                    {~d.money.amount: 0.0}, "write"))
                updates.append(ForwardUpdate(self.state.history,
                    _onehot_hist(d, next_hist), "write"))
                updates.append(ForwardUpdate(self.state.phase,
                    _onehot_phase(d, "terminal"), "write"))

                # Reward to MS: agent's final stack delta from hand-start
                # (positive = won money this hand, negative = lost)
                final_agent_stack = (new_winner_stack if winner_p1
                                     else (agent_stack - raise_amount if (action in ("min_raise","max_raise") and actor == "p1")
                                           else agent_stack - last_raise if (action == "call" and actor == "p1")
                                           else agent_stack))

            chip_delta = final_agent_stack - (self._stack_at_hand_start or STARTING_STACK)
            updates.append(ForwardUpdate(self.ms_chip_in,
                {~b.main.rwd * ~d.reward.main: chip_delta}, "write"))

            # Log
            if self.log_callback is not None:
                self.log_callback({
                    "actions": list(self.actions_seq),
                    "reward_p1": chip_delta,
                    "p1_card": _read_oh(self.state.card_p1[0], d.card, ["j","q","k"]),
                    "p2_card": _read_oh(self.state.card_p2[0], d.card, ["j","q","k"]),
                    "pot_won": (0.0 if is_tie else pot_after_action),
                    "winner": ("tie" if is_tie else ("p1" if winner_p1 else "p2")),
                    "ended_by": "fold" if action == "fold" else "showdown",
                })
            self.actions_seq.clear()
            self._stack_at_hand_start = None

            # Schedule next hand
            self.system.schedule(Event(
                source=self._restart,
                updates=[ForwardUpdate(self.state.phase,
                    _onehot_phase(d, "hand_start"), "write")],
                time=timedelta(milliseconds=1),
                priority=Priority.DEFERRED,
            ))
        else:
            # Non-terminal: advance to next history + switch phase
            updates.append(ForwardUpdate(self.state.history,
                _onehot_hist(d, next_hist), "write"))
            next_phase_name = ("opponent_turn" if actor == "p1"
                               else "agent_turn")
            updates.append(ForwardUpdate(self.state.phase,
                _onehot_phase(d, next_phase_name), "write"))
            # If agent's turn, push perception
            if actor == "p2":
                card_p1_name = _read_oh(self.state.card_p1[0], d.card, ["j","q","k"])
                updates.append(ForwardUpdate(self.agent_ipt.main, {
                    ~b.main.wm * ~_card_atom(d, card_p1_name): 1.0,
                    ~b.main.wm * ~_hist_atom(d, next_hist): 1.0,
                }, "push"))
                # Trigger the agent's next decision (after perception lands)
                if self.selector is not None:
                    self.system.schedule(self.selector.trigger(
                        dt=timedelta(microseconds=1)))

        return Event(self.advance, updates, dt, priority)

    def _restart(self): pass

    @staticmethod
    def _after_check(seq, actor):
        """What history follows a check action?"""
        # seq just got 'check' appended
        if seq == ["check"]:                  return "p1_checked"   # P1 just checked
        if seq == ["check", "check"]:         return "terminal_showdown"
        return "terminal_showdown"

    @staticmethod
    def _after_raise(seq, actor, action):
        size = "min_raise" if action == "min_raise" else "max_raise"
        if actor == "p1" and len(seq) == 1:
            return f"p1_{size}"
        if actor == "p2":
            # P2 raised after P1 checked
            return f"check_{size}"
        # P1 raised after own check + P2 raised (re-raise) — not modeled in v2
        raise RuntimeError(f"unexpected raise: seq={seq}, actor={actor}")


# ─── Read helpers ────────────────────────────────────────────────────────
def _read_oh(numdict, sort, names):
    for n in names:
        if numdict[~getattr(sort, n)] > 0.5:
            return n
    raise RuntimeError(f"No one-hot active in {sort}")


def _read_hist(numdict, d):
    names = ["start", "p1_checked", "p1_min_raise", "p1_max_raise",
             "check_min_raise", "check_max_raise"]
    for n in names:
        if numdict[~getattr(d.history, n)] > 0.5:
            return n
    raise RuntimeError("No history one-hot active")


def _read_action(numdict, d, actor):
    for action_name in ["check", "min_raise", "max_raise", "call", "fold"]:
        key = ~getattr(d.player, actor) * ~getattr(d.action, action_name)
        if numdict[key] > 0.5:
            return action_name
    raise RuntimeError(f"No action read for {actor}")
