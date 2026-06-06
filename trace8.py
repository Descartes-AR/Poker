"""Traced version of check 8 — logs every event and dumps state at the stall."""
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from keyspace import PokerKeyspace
from simulation import PokerSimulation
from knowledge import initialize_all_knowledge
from opponents import calling_station

root = PokerKeyspace()
hand_log = []
sim = PokerSimulation("sim_test", root,
    opponent_policy=calling_station, hand_log=hand_log, seed=0)

print(">>> initializing knowledge")
initialize_all_knowledge(sim, sd=0.1)

d = root.d

def dump_state(tag):
    def oh(site, names, sort):
        active = [n for n in names if site[0][~getattr(sort, n)] > 0.5]
        return active or ["<none>"]
    print(f"    [{tag}] phase={oh(sim.state.phase, ['hand_start','agent_turn','opponent_turn','terminal','bust'], d.phase)}"
          f" hist={oh(sim.state.history, ['start','p1_checked','p1_min_raise','p1_max_raise','check_min_raise','check_max_raise','terminal_showdown','terminal_fold'], d.history)}"
          f" last_actor={oh(sim.state.last_actor, ['p1','p2'], d.player)}")
    at = sim.state.action_taken[0]
    acts = []
    for pl in ['p1','p2']:
        for ac in ['check','min_raise','max_raise','call','fold']:
            if at[~getattr(d.player,pl) * ~getattr(d.action,ac)] > 0.5:
                acts.append(f"{pl}:{ac}")
    print(f"         action_taken={acts or ['<none>']}"
          f" pot={sim.state.pot[0][~d.money.amount]}"
          f" a_stack={sim.state.agent_stack[0][~d.money.amount]}")
    # selector output
    try:
        sel = sim.acs.selector.main[0]
        chosen = [ac for ac in ['check','min_raise','max_raise','call','fold']
                  if sel[~root.b.main.wm * ~getattr(d.action,ac)] > 0.5]
        print(f"         selector.main chosen={chosen or ['<none>']}")
    except Exception as e:
        print(f"         selector read err: {e}")

print(">>> kickoff")
sim.kickoff_first_hand()
dump_state("after kickoff, before run")

print(">>> running events")
n = 0
for event in sim.run():
    n += 1
    src = getattr(event, "source", None)
    src_name = getattr(src, "__name__", None) or getattr(src, "__qualname__", str(src))
    print(f"  event {n}: source={src_name}")
    if n >= 60 or len(hand_log) >= 1:
        break

print(f">>> stopped after {n} events; hands logged={len(hand_log)}")
dump_state("final")
