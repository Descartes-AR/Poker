"""Instrument the selector's update to see if the BackwardUpdate error is non-empty,
and watch whether the Pool's resolve ever sees a gradient on its main."""
from keyspace import PokerKeyspace
from simulation import PokerSimulation
from knowledge import initialize_all_knowledge
from opponents import calling_station
from pyClarion.events import BackwardUpdate, ForwardUpdate

root = PokerKeyspace()
sim = PokerSimulation("g", root, opponent_policy=calling_station, hand_log=[], seed=0)
initialize_all_knowledge(sim, sd=0.1)

sel = sim.acs.selector
pool = sim.acs.pool
bottom = sim.acs.bottom

# Wrap selector.update to inspect the error it emits
orig_update = sel.update
upd_stats = {"calls": 0, "empty_error": 0, "nonempty_error": 0,
             "targets_pool_main": 0}
def wrapped_update(dt=None, priority=None):
    ev = orig_update() if dt is None else orig_update(dt, priority)
    upd_stats["calls"] += 1
    for u in ev.updates:
        if isinstance(u, BackwardUpdate):
            data = getattr(u, "data", None)
            n = len(data.d) if (data is not None and hasattr(data, "d")) else (len(data) if data is not None else 0)
            if n == 0:
                upd_stats["empty_error"] += 1
            else:
                upd_stats["nonempty_error"] += 1
            # is the target the pool.main?
            tgt = getattr(u, "state", None) or getattr(u, "site", None)
            if tgt is pool.main:
                upd_stats["targets_pool_main"] += 1
    return ev
sel.update = wrapped_update

# Also check selector tape/reward state at first few updates
orig_pool_resolve = pool.resolve
pool_stats = {"resolve_calls": 0, "saw_main_grad": 0, "tape_full": 0}
def wrapped_pool_resolve(event):
    pool_stats["resolve_calls"] += 1
    grads = event.index(BackwardUpdate)
    if pool.main in grads:
        pool_stats["saw_main_grad"] += 1
    if len(pool.tapes) == pool.tapes.maxlen:
        pool_stats["tape_full"] += 1
    return orig_pool_resolve(event)
pool.resolve = wrapped_pool_resolve

sim.kickoff_first_hand()
hl = sim.hand_log
for _ in sim.run():
    if len(hl) >= 50:
        break

print("=== selector.update stats ===")
print(upd_stats)
print()
print("=== pool.resolve stats ===")
print(pool_stats)
print()
print("pool.tapes maxlen:", pool.tapes.maxlen, " len now:", len(pool.tapes))
print("bottom.tapes maxlen:", bottom.tapes.maxlen, " len now:", len(bottom.tapes))
print()

# Inspect selector internal tapes/reward at end
print("=== selector internal state (deque lengths) ===")
for attr in ["qvals", "actions", "reward", "cost"]:
    st = getattr(sel, attr, None)
    if st is not None and hasattr(st, "data"):
        try:
            print(f"  {attr}.data len = {len(st.data)} (maxlen {st.data.maxlen})")
        except Exception as e:
            print(f"  {attr}: {e}")

# Try to read what the error would be right now
print()
print("=== current selector.reward contents (last few) ===")
try:
    for i, r in enumerate(sel.reward.data):
        print(f"  reward[t-{i}]:", dict(r.d) if hasattr(r,'d') else r)
except Exception as e:
    print("  err:", e)
