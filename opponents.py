"""
Opponent policies for v2 no-limit single-card poker.

Each policy takes a context dict:
    {card: 'j'|'q'|'k', history: <str>,
     pot: float, own_stack: float, opp_stack: float, last_raise: float}
and returns one of: 'check', 'min_raise', 'max_raise', 'call', 'fold'
"""

import random


def calling_station(ctx):
    """Never folds, never raises. Just checks or calls."""
    facing_bet = ctx["history"] in ("p1_min_raise", "p1_max_raise",
                                     "check_min_raise", "check_max_raise")
    return "call" if facing_bet else "check"


def rock(ctx):
    """Only bets/calls with K, folds Q/J to any raise."""
    is_strong = (ctx["card"] == "k")
    facing_bet = ctx["history"] in ("p1_min_raise", "p1_max_raise",
                                     "check_min_raise", "check_max_raise")
    if facing_bet:
        return "call" if is_strong else "fold"
    return "max_raise" if is_strong else "check"


_nash_rng = random.Random(42)


def loose_aggressive(ctx):
    """Calls with anything, raises with K and J (bluffs occasionally).

    More interesting opponent than the static ones — this opponent itself
    bluffs, creating an environment where the agent needs to learn to
    occasionally call with medium cards too.
    """
    facing_bet = ctx["history"] in ("p1_min_raise", "p1_max_raise",
                                     "check_min_raise", "check_max_raise")
    if facing_bet:
        # Call most of the time, fold J vs max_raise
        if ctx["card"] == "j" and ctx["history"] in ("p1_max_raise",
                                                       "check_max_raise"):
            return "fold" if _nash_rng.random() < 0.7 else "call"
        return "call"
    # First-to-act: K always raises big, J bluffs sometimes, Q checks
    if ctx["card"] == "k": return "max_raise"
    if ctx["card"] == "j":
        return "min_raise" if _nash_rng.random() < 0.3 else "check"
    return "check"
