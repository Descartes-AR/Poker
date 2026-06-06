"""
Keyspace for no-limit single-card poker (Kuhn variant with variable bet sizing).

CHANGES from the standard Kuhn version:
  - Action atoms: 5 (check, min_raise, max_raise, call, fold)
  - History atoms: expanded for raise-size contexts
  - Money sort: scalar holder for pot/stack sites
"""

from pyClarion.knowledge import (Root, BusFamily, Buses, Bus,
                                  AtomFamily, ChunkFamily, DataFamily,
                                  Atoms, Atom)


# ─── Feature atoms ───────────────────────────────────────────────────────
class Card(Atoms):
    j: Atom; q: Atom; k: Atom


class History(Atoms):
    """Reachable decision contexts in the v2 game tree.

    Game tree (one raise per player max, no re-raises):
        start              → P1 first action: {check, min_raise, max_raise}
        p1_checked         → P2 acts after P1 check: {check, min_raise, max_raise}
        p1_min_raise       → P2 facing P1 small-raise: {call, fold}
        p1_max_raise       → P2 facing P1 big-raise:   {call, fold}
        check_min_raise    → P1 acts after own check + P2 small-raise: {call, fold}
        check_max_raise    → P1 acts after own check + P2 big-raise:   {call, fold}
    Plus terminal markers for logging.
    """
    start: Atom
    p1_checked: Atom
    p1_min_raise: Atom
    p1_max_raise: Atom
    check_min_raise: Atom
    check_max_raise: Atom
    terminal_showdown: Atom
    terminal_fold: Atom


class Player(Atoms):
    p1: Atom
    p2: Atom


class Phase(Atoms):
    hand_start: Atom
    agent_turn: Atom
    opponent_turn: Atom
    terminal: Atom
    bust: Atom            # a player ran out of money


class Action(Atoms):
    """Five-element action space. Legality depends on history.

      check     : legal only when no bet pending (start, p1_checked)
      min_raise : legal when no bet pending (raise = half pot)
      max_raise : legal when no bet pending (raise = full pot)
      call      : legal only when facing a bet (p1_*_raise, check_*_raise)
      fold      : legal only when facing a bet
    """
    check: Atom
    min_raise: Atom
    max_raise: Atom
    call: Atom
    fold: Atom


# ─── NACS concepts ───────────────────────────────────────────────────────
class Strength(Atoms):
    strong: Atom
    medium: Atom
    weak: Atom


# ─── Motivational ────────────────────────────────────────────────────────
class Drive(Atoms):
    profit: Atom


class Reward(Atoms):
    main: Atom


# ─── Money (scalar holder for pot/stack sites) ───────────────────────────
class Money(Atoms):
    """Single atom holder so pot/stack sites have a valid keyform."""
    amount: Atom


# ─── Buses ───────────────────────────────────────────────────────────────
class MainBuses(Buses):
    wm: Bus       # working memory: features the agent perceives
    nacs: Bus     # NACS strength activations
    drive: Bus    # MS drive state
    rwd: Bus      # reward signal
    world: Bus    # game state (cards, phase, action_taken, last_actor)
    money: Bus    # pot + stacks


class PokerLayout(BusFamily):
    main: MainBuses


class PokerData(DataFamily):
    card: Card
    history: History
    player: Player
    phase: Phase
    action: Action
    strength: Strength
    drive: Drive
    reward: Reward
    money: Money


class PokerKeyspace(Root):
    b: PokerLayout
    d: PokerData
    p: AtomFamily
    s: AtomFamily
    k_acs: ChunkFamily
    k_nacs: ChunkFamily
