"""Knowledge init for v2: encode chunks, set rule weights for 5-action recs,
randomize bottom-layer weights."""

import random
from datetime import timedelta

from pyClarion import Priority, Event
from pyClarion.events import ForwardUpdate

from nacs import nacs_chunk_defs
from acs import acs_chunk_defs, RULE_RECOMMENDATIONS


def initialize_all_knowledge(sim, sd=0.1):
    nacs_defs = nacs_chunk_defs(sim.root)
    sim.system.schedule(sim.nacs.chunks.encode(*nacs_defs))
    _drain(sim)

    acs_defs = acs_chunk_defs(sim.root)
    sim.system.schedule(sim.acs.chunks.encode(*acs_defs))
    _drain(sim)

    _set_rule_weights(sim, acs_defs)
    _drain(sim)

    _init_bottom_weights(sim, sd=sd)
    _drain(sim)


def _drain(sim):
    for _ in sim.run(): pass


def _set_rule_weights(sim, acs_chunks):
    """Write rule layer weights based on RULE_RECOMMENDATIONS.

    For each (chunk, recommended_action) pair: +1 to that action key,
    -0.25 to all other actions (so the recommendation isn't catastrophically
    against alternatives — leaves room for the bottom level to disagree).
    """
    d, b = sim.root.d, sim.root.b
    wm = b.main.wm
    action_atoms = {
        "check":     d.action.check,
        "min_raise": d.action.min_raise,
        "max_raise": d.action.max_raise,
        "call":      d.action.call,
        "fold":      d.action.fold,
    }

    weight_updates = {}
    for chunk in acs_chunks:
        rec_name = RULE_RECOMMENDATIONS[chunk._name_]
        chunk_key = ~chunk
        for action_name, atom in action_atoms.items():
            action_key = ~wm * ~atom
            w = +1.0 if action_name == rec_name else -0.25
            weight_updates[chunk_key * action_key] = w

    sim.system.schedule(Event(
        source=_set_rule_weights,
        updates=[ForwardUpdate(sim.acs.rules.weights, weight_updates, "write")],
        time=timedelta(), priority=Priority.LEARNING,
    ))


def _init_bottom_weights(sim, sd=0.1):
    rng = random.Random(0)
    layer = sim.acs.bottom
    updates = []
    for site in [layer.weights, layer.bias]:
        new_vals = {key: rng.gauss(0.0, sd) for key in site.index}
        updates.append(ForwardUpdate(site, new_vals, "write"))
    sim.system.schedule(Event(
        source=_init_bottom_weights, updates=updates,
        time=timedelta(), priority=Priority.LEARNING,
    ))
