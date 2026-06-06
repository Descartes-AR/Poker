"""
Stepwise smoke test for the Clarion Kuhn-poker simulation.

Runs eight self-contained checks, each in isolation. Each check prints its
result and fails fast on the first error. Run with:
    python debug.py
or run a single check:
    python debug.py 3    # only check #3

The checks are ordered from least-dependent to most-dependent. If check N
fails, checks N+1..8 will likely fail too — fix N first.

Check 0: Imports
Check 1: Keyspace construction
Check 2: GameState — sites hold values
Check 3: NACS — chunk encoding + bottom-up activation
Check 4: ACS — chunk encoding + rule layer weight writing
Check 5: Pool — combination of two layer outputs
Check 6: Selector — TDLearning trigger and action read
Check 7: One Dealer→action cycle without opponent
Check 8: Full first hand end-to-end (one hand, one opponent)
"""

import sys
import traceback


def check(num, name):
    """Decorator: prints header before, captures exceptions, returns bool."""
    def deco(fn):
        def wrapped():
            print(f"\n{'='*60}")
            print(f"CHECK {num}: {name}")
            print('='*60)
            try:
                fn()
                print(f"  [PASS] check {num}")
                return True
            except Exception as e:
                print(f"  [FAIL] check {num}: {type(e).__name__}: {e}")
                traceback.print_exc()
                return False
        wrapped._num = num
        wrapped._name = name
        return wrapped
    return deco


# ─── Check 0 ─────────────────────────────────────────────────────────────
@check(0, "Imports — pyClarion + our modules")
def check_imports():
    import pyClarion
    print(f"  pyClarion at {pyClarion.__file__}")
    from pyClarion import (Agent, Input, Layer, Pool, ChunkStore,
                            Event, Priority)
    from pyClarion.components.learning import TDLearning
    from pyClarion.components.optimizers import Adam
    from pyClarion.events import ForwardUpdate
    print("  pyClarion symbols imported OK")

    import keyspace, game, nacs, acs, ms, mcs, simulation, knowledge, opponents
    print("  project modules imported OK")


# ─── Check 1 ─────────────────────────────────────────────────────────────
@check(1, "Keyspace — construct a Root, address each subsystem")
def check_keyspace():
    from keyspace import PokerKeyspace
    root = PokerKeyspace()
    # Address every family slot
    print(f"  b.main.wm   = {root.b.main.wm}")
    print(f"  b.main.world= {root.b.main.world}")
    print(f"  b.main.money= {root.b.main.money}")
    print(f"  d.card      = {root.d.card}")
    print(f"  d.card.j    = {root.d.card.j}")
    print(f"  d.history   = {root.d.history}")
    print(f"  d.action    = {root.d.action}")
    print(f"  d.money.amount = {root.d.money.amount}")
    print(f"  p (param family) = {root.p}")
    print(f"  s (state family) = {root.s}")
    print(f"  k_acs (chunk family) = {root.k_acs}")
    # Action enumeration — make sure all 5 exist
    actions = [root.d.action.check, root.d.action.min_raise,
               root.d.action.max_raise, root.d.action.call,
               root.d.action.fold]
    print(f"  5 action atoms: {[str(a) for a in actions]}")


# ─── Check 2 ─────────────────────────────────────────────────────────────
@check(2, "GameState — instantiate inside an Agent, sites hold values")
def check_gamestate():
    from pyClarion import Agent
    from keyspace import PokerKeyspace
    from game import GameState, STARTING_STACK

    root = PokerKeyspace()
    agent = Agent("check2_agent", root)
    with agent:
        gs = GameState("test_state", root)

    # All sites should exist and be readable
    print(f"  card_p1[0] keys: {len(gs.card_p1[0].d)}")
    print(f"  history[0] keys: {len(gs.history[0].d)}")
    print(f"  phase[0] keys:   {len(gs.phase[0].d)}")
    print(f"  pot[0] = {gs.pot[0].d}")
    print(f"  agent_stack[0] = {gs.agent_stack[0].d}")
    print(f"  opp_stack[0] = {gs.opp_stack[0].d}")

    # Initial values: pot=0, stacks=STARTING_STACK
    d = root.d
    pot_val = gs.pot[0][~d.money.amount]
    agent_val = gs.agent_stack[0][~d.money.amount]
    print(f"  initial pot = {pot_val}, agent_stack = {agent_val}")
    assert agent_val == STARTING_STACK, \
        f"expected agent_stack=={STARTING_STACK}, got {agent_val}"


# ─── Check 3 ─────────────────────────────────────────────────────────────
@check(3, "NACS — chunks encode, bottom-up activates from card features")
def check_nacs():
    from pyClarion import Agent, Input
    from keyspace import PokerKeyspace
    from nacs import NACSModule, nacs_chunk_defs

    root = PokerKeyspace()
    agent = Agent("check3_agent", root)
    b, d = root.b, root.d
    with agent:
        ipt = Input("test_ipt", (b.main.wm, d))
        nacs_m = NACSModule("test_nacs", root)
        ipt >> nacs_m.bu

    # Encode chunks
    defs = nacs_chunk_defs(root)
    print(f"  encoding {len(defs)} NACS chunks")
    agent.system.schedule(nacs_m.chunks.encode(*defs))
    n_events = 0
    for ev in agent.run():
        n_events += 1
        if n_events > 100: break
    print(f"  drained {n_events} events after encode")

    # Send card.k as input — expect strong_hand chunk to activate
    print("  sending card.k = 1.0 to input")
    agent.system.schedule(ipt.send({~b.main.wm * ~d.card.k: 1.0}))
    n_events = 0
    for ev in agent.run():
        n_events += 1
        if n_events > 100: break
    print(f"  drained {n_events} events after input send")

    # Read bottom-up output — should have strong_hand near 1
    bu_out = nacs_m.bu.main[0]
    print(f"  bu.main[0] entries: {dict(bu_out.d)}")
    # Find which chunk has highest activation
    if bu_out.d:
        top_key, top_val = max(bu_out.d.items(), key=lambda kv: kv[1])
        print(f"  most-activated chunk: {top_key} = {top_val:.3f}")
        # Should mention 'strong' in key string
        assert "strong" in str(top_key).lower(), \
            f"expected strong_hand chunk active for card.k, got {top_key}"


# ─── Check 4 ─────────────────────────────────────────────────────────────
@check(4, "ACS — chunks encode + rule weights write + activation flows")
def check_acs():
    from pyClarion import Agent, Input
    from keyspace import PokerKeyspace
    from nacs import NACSModule
    from acs import ACSModule, acs_chunk_defs, RULE_RECOMMENDATIONS

    root = PokerKeyspace()
    agent = Agent("check4_agent", root)
    b, d = root.b, root.d
    with agent:
        ipt = Input("test_ipt", (b.main.wm, d))
        nacs_m = NACSModule("test_nacs", root)
        acs_m = ACSModule("test_acs", root, nacs_m, lr=1e-2)
        ipt >> nacs_m.bu
        ipt >> acs_m.bu
        ipt >> acs_m.bottom

    # Encode ACS chunks
    defs = acs_chunk_defs(root)
    print(f"  encoding {len(defs)} ACS chunks")
    agent.system.schedule(acs_m.chunks.encode(*defs))
    for _ in agent.run(): pass

    # Try writing rule weights — this is the part most likely to break
    from pyClarion import Event, Priority
    from pyClarion.events import ForwardUpdate
    from datetime import timedelta

    wm = b.main.wm
    action_atoms = {
        "check": d.action.check, "min_raise": d.action.min_raise,
        "max_raise": d.action.max_raise, "call": d.action.call,
        "fold": d.action.fold,
    }
    weight_updates = {}
    for chunk in defs:
        rec_name = RULE_RECOMMENDATIONS[chunk._name_]
        chunk_key = ~chunk
        for action_name, atom in action_atoms.items():
            action_key = ~wm * ~atom
            w = +1.0 if action_name == rec_name else -0.25
            weight_updates[chunk_key * action_key] = w

    print(f"  writing {len(weight_updates)} weight entries to rule layer")
    def src(): pass
    agent.system.schedule(Event(
        source=src, updates=[ForwardUpdate(acs_m.rules.weights,
            weight_updates, "write")],
        time=timedelta(), priority=Priority.LEARNING))
    for _ in agent.run(): pass

    # Verify: rule weights exist
    actual_w = acs_m.rules.weights[0].d
    print(f"  rule layer weights count: {len(actual_w)}")
    if actual_w:
        # Print a sample
        sample = next(iter(actual_w.items()))
        print(f"  sample weight: {sample[0]} = {sample[1]:.3f}")


# ─── Check 5 ─────────────────────────────────────────────────────────────
@check(5, "Pool — combine two layer outputs (top and bottom)")
def check_pool():
    from pyClarion import Agent, Input, Pool, Layer
    from keyspace import PokerKeyspace
    root = PokerKeyspace()
    agent = Agent("check5_agent", root)
    b, d, p = root.b, root.d, root.p
    feat_d = (b.main.wm, d)
    action_d = (b.main.wm, d.action)
    with agent:
        ipt = Input("test_ipt", feat_d)
        layer_a = ipt >> Layer("layer_a", i=feat_d, o=action_d, l=2)
        layer_b = ipt >> Layer("layer_b", i=feat_d, o=action_d, l=2)
        pool = Pool("test_pool", p, action_d)
        (layer_a, layer_b) >> pool

    # Inspect pool params — should have two per-input weight atoms
    pool_params = dict(pool.params[0].d)
    print(f"  pool.params[0].d: {pool_params}")
    print(f"  pool has {len(pool_params)} input weights")
    assert len(pool_params) == 2, \
        f"expected 2 pool inputs, got {len(pool_params)}"


# ─── Check 6 ─────────────────────────────────────────────────────────────
@check(6, "TDLearning Selector — trigger + read chosen action")
def check_selector():
    from pyClarion import Agent, Input, Layer, Pool
    from pyClarion.components.learning import TDLearning
    from keyspace import PokerKeyspace
    root = PokerKeyspace()
    agent = Agent("check6_agent", root)
    b, d, p, s = root.b, root.d, root.p, root.s
    feat_d = (b.main.wm, d)
    action_d = (b.main.wm, d.action)
    reward_d = (b.main.rwd, d.reward)
    with agent:
        ipt = Input("ipt", feat_d)
        layer = ipt >> Layer("bot", i=feat_d, o=action_d, l=2)
        pool = Pool("pool", p, action_d)
        layer >> pool
        sel = pool >> TDLearning("sel", p=p, s=s, d=action_d, r=reward_d,
                                  gamma=0.9, sd=0.5, f=0.0, l=2)

    # Send input + trigger
    print("  sending input (card.j) and triggering")
    agent.system.schedule(ipt.send({~b.main.wm * ~d.card.j: 1.0}))
    agent.system.schedule(sel.trigger())
    n_events = 0
    for _ in agent.run():
        n_events += 1
        if n_events > 200: break
    print(f"  drained {n_events} events")

    # Read what got selected
    main = sel.main[0]
    print(f"  selector.main[0]: {dict(main.d)}")
    # poll() returns dict[Key, Key]
    polled = sel.poll()
    print(f"  selector.poll(): {polled}")


# ─── Check 7 ─────────────────────────────────────────────────────────────
@check(7, "Dealer — deals cards, sets phase, pushes perception")
def check_dealer():
    from pyClarion import Agent, Input
    from keyspace import PokerKeyspace
    from game import GameState, Dealer, BLIND, INITIAL_POT

    root = PokerKeyspace()
    agent = Agent("check7_agent", root)
    d = root.d
    b = root.b
    with agent:
        ipt = Input("ipt", (b.main.wm, d))
        gs = GameState("gs", root)
        dealer = Dealer("dealer", root, gs, ipt, seed=42)

    # Trigger hand_start
    from pyClarion import Event, Priority
    from pyClarion.events import ForwardUpdate
    from datetime import timedelta
    def src(): pass
    agent.system.schedule(Event(source=src, updates=[
        ForwardUpdate(gs.phase, {~d.phase.hand_start: 1.0}, "write")
    ], time=timedelta(), priority=Priority.PROPAGATION))

    n = 0
    for _ in agent.run():
        n += 1
        if n > 100: break
    print(f"  drained {n} events after kickoff")

    # Check state
    pot = gs.pot[0][~d.money.amount]
    a_stack = gs.agent_stack[0][~d.money.amount]
    o_stack = gs.opp_stack[0][~d.money.amount]
    print(f"  after deal: pot={pot}, agent_stack={a_stack}, opp_stack={o_stack}")
    assert pot == INITIAL_POT, f"pot should be {INITIAL_POT}, got {pot}"
    assert a_stack == 150 - BLIND, f"agent_stack should be {150-BLIND}"

    # Phase should now be agent_turn
    print(f"  phase: agent_turn? {gs.phase[0][~d.phase.agent_turn]}")
    assert gs.phase[0][~d.phase.agent_turn] > 0.5, \
        "phase should be agent_turn after deal"

    # Agent's input should have card+history.start
    ipt_main = ipt.main[0].d
    print(f"  agent input keys after deal: {list(ipt_main.keys())[:5]}...")
    # Should have history.start active
    assert ipt_main.get(~b.main.wm * ~d.history.start, 0.0) > 0.5, \
        "agent input should have history.start active"


# ─── Check 8 ─────────────────────────────────────────────────────────────
@check(8, "Full one-hand round-trip end-to-end")
def check_full_hand():
    from keyspace import PokerKeyspace
    from simulation import PokerSimulation
    from knowledge import initialize_all_knowledge
    from opponents import calling_station

    root = PokerKeyspace()
    hand_log = []
    sim = PokerSimulation("sim_test", root,
        opponent_policy=calling_station,
        hand_log=hand_log, seed=0)

    print("  initializing knowledge ...")
    initialize_all_knowledge(sim, sd=0.1)

    print("  kicking off first hand ...")
    sim.kickoff_first_hand()

    n_events = 0
    n_hands_target = 3
    for _ in sim.run():
        n_events += 1
        if len(hand_log) >= n_hands_target:
            break
        if n_events > 10000:
            print(f"  [WARN] {n_events} events with only {len(hand_log)} hands "
                  f"logged — possible deadlock?")
            break

    print(f"  events processed: {n_events}")
    print(f"  hands logged:     {len(hand_log)}")
    if hand_log:
        print(f"  first hand: {hand_log[0]}")
    assert len(hand_log) >= 1, "should log at least one hand"


# ─── Main ────────────────────────────────────────────────────────────────
ALL_CHECKS = [check_imports, check_keyspace, check_gamestate,
              check_nacs, check_acs, check_pool, check_selector,
              check_dealer, check_full_hand]


def main(argv):
    if len(argv) > 1:
        # Run just one check
        n = int(argv[1])
        ALL_CHECKS[n]()
    else:
        # Run all, stop at first failure
        for fn in ALL_CHECKS:
            ok = fn()
            if not ok:
                print(f"\n[STOPPED] check {fn._num} failed. Fix that first.")
                return 1
        print(f"\n{'='*60}\n[ALL CHECKS PASSED]\n{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
