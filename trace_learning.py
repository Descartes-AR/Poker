"""Trace the learning event chain: do backward() and optimizer.update() ever fire?"""
from keyspace import PokerKeyspace
from simulation import PokerSimulation
from knowledge import initialize_all_knowledge
from opponents import calling_station

root = PokerKeyspace()
hand_log = []
sim = PokerSimulation("trace", root, opponent_policy=calling_station,
                      hand_log=hand_log, seed=0)
initialize_all_knowledge(sim, sd=0.1)

# Count occurrences of each event source name
from collections import Counter
counts = Counter()

# Identify the key sources we care about
bottom_backward = sim.acs.bottom.backward
opt_update = sim.acs.optimizer.update
sel_update = sim.acs.selector.update
sel_select = sim.acs.selector.select

# Check tape maxlen
print("bottom.tapes maxlen:", sim.acs.bottom.tapes.maxlen)
print("bottom.tapes current len:", len(sim.acs.bottom.tapes))

sim.kickoff_first_hand()
matched_backward = 0
matched_optupdate = 0
n = 0
for event in sim.run():
    n += 1
    src = getattr(event, "source", None)
    name = getattr(src, "__qualname__", None) or getattr(src, "__name__", str(src))
    counts[name] += 1
    if src == bottom_backward: matched_backward += 1
    if src == opt_update: matched_optupdate += 1
    if len(hand_log) >= 50:
        break

print(f"\nprocessed {n} events over {len(hand_log)} hands")
print(f"bottom.tapes len after run: {len(sim.acs.bottom.tapes)}")
print(f"\n=== event source counts (top 25) ===")
for name, c in counts.most_common(25):
    print(f"  {c:5d}  {name}")

print(f"\n=== KEY MATCHES ===")
print(f"  events where source == bottom.backward:    {matched_backward}")
print(f"  events where source == optimizer.update:   {matched_optupdate}")
print(f"\n  (if bottom.backward count is 0 → backward() never fires → upstream problem)")
print(f"  (if backward fires but optimizer.update is 0 → my trigger condition is wrong)")

# Also check: did weights move?
w = sim.acs.bottom.weights[0].d
import numpy as np
print(f"\n  bottom.weights L2 norm now: {np.sqrt(sum(v*v for v in w.values())):.4f}")
print(f"  (initial random sd=0.1 over 145 entries → expect ~1.2 if untouched)")

# Check if grad sites have accumulated anything
try:
    g = sim.acs.bottom.weights.grad[0].d
    print(f"  bottom.weights.grad nonzero entries: {sum(1 for v in g.values() if abs(v)>1e-12)}")
    print(f"  bottom.weights.grad L2: {np.sqrt(sum(v*v for v in g.values())):.6f}")
    print(f"  (if grad is NONZERO and accumulating → optimizer never consumed it)")
except Exception as e:
    print(f"  (could not read .grad: {e})")
