"""
ACS module — v3 (LAST VERSION THAT RUNS trace_learning.py WITHOUT A TRACEBACK).

This is a deliberate checkpoint for study. Relative to the final (v5) acs.py,
this version OMITS the two changes that introduced the tracebacks:

  REMOVED (was change #4): the `pool.forward()` trigger in resolve().
      Adding it made the Pool actually run its forward pass, which on
      Python 3.14 raised `ValueError: no signature found for builtin CAM`
      inside the gradient tape.

  REMOVED (was change #5): the `_patch_cam_signature()` shim.
      That shim fixed the signature error, but then exposed the deeper issue:
      `cam.grad(...)` raises NotImplementedError — the cam aggregator is not
      differentiable in pyClarion, so gradients cannot flow back through the
      Pool.

  KEPT (change #2): the optimizer trigger (optimizer.update on bottom.backward).
  KEPT (change #3): Pool l=2.

CONSEQUENCE: this version runs cleanly but DOES NOT LEARN. Because nothing
triggers pool.forward(), the Pool never combines its inputs, never builds a
tape, and its backward never fires — so the cam gradient code is never reached
(no error) and the bottom-layer weights never change (no learning). When you
run trace_learning.py against this file you should see:
    pool.forward calls : 0
    bottom.backward    : 0
    optimizer.update   : 0
    weight L2 change   : 0.000000
...and NO traceback.

This is the cleanest place to study what introduced the errors: the tracebacks
appear the instant you reconnect the learning path (trigger pool.forward), which
reveals that cam is non-differentiable. The fix for actual learning is NOT to
re-add these two changes as-is, but to either (Option 1) train the bottom layer
on a path that bypasses the cam Pool, or (Option 3) give cam a straight-through
gradient. The Pool/cam stays as the action-SELECTION combiner either way.
"""

from pyClarion import Layer, Pool, ChunkStore
from pyClarion.components.learning import TDLearning
from pyClarion.components.optimizers import Adam
from pyClarion.components.base import Component

from keyspace import PokerKeyspace


class ACSModule(Component):
    def __init__(self, name, root, nacs_module,
                 *, lr=1e-2, gamma=0.9, sd=0.5):
        super().__init__(name)
        self.root = root
        b, d = root.b, root.d
        p, s, k = root.p, root.s, root.k_acs
        feat_d   = (b.main.wm, d)
        action_d = (b.main.wm, d.action)
        reward_d = (b.main.rwd, d.reward)

        with self:
            # ── ACS top: chunks + fixed rule layer ──
            self.chunks = ChunkStore(f"{name}.chunks", c=k, d=feat_d)
            self.bu = self.chunks.bottom_up(f"{name}.bu")
            self.rules = self.bu >> Layer(
                f"{name}.rules", i=self.chunks.c, o=action_d, l=2)

            # ── ACS bottom: trainable Q-network ──
            self.bottom = Layer(
                f"{name}.bottom", i=feat_d, o=action_d, l=2)

            # ── Combination ──
            # l=2 kept (change #3). The default l=1 would overwrite the tape
            # before any backward arrives; l=2 matches the TD two-step window.
            self.pool = Pool(f"{name}.pool", p, action_d, l=2)
            (self.bottom, self.rules) >> self.pool

            # ── Selection + learning ──
            self.selector = self.pool >> TDLearning(
                f"{name}.selector",
                p=p, s=s, d=action_d, r=reward_d,
                gamma=gamma, sd=sd, f=0.0, l=2)

            # ── Optimizer (BOTTOM only) ──
            self.optimizer = Adam(f"{name}.optimizer", p, lr=lr)
            self.optimizer.add(self.bottom.weights, self.bottom.bias)

    def resolve(self, event):
        # Optimizer trigger kept (change #2). In THIS version it never actually
        # fires, because bottom.backward never fires (the Pool never forwards,
        # so no gradient ever reaches the bottom layer). It is harmless here and
        # is the correct mechanism once a differentiable learning path exists.
        if event.source == self.bottom.backward:
            self.system.schedule(self.optimizer.update())


def acs_chunk_defs(root: PokerKeyspace):
    """18 situation chunks: 3 card strengths × 6 history contexts."""
    d = root.d
    wm = root.b.main.wm
    defs = []
    strength_to_card = {"strong": d.card.k, "medium": d.card.q, "weak": d.card.j}
    for strength_name, card_atom in strength_to_card.items():
        for hist_name in ["start", "p1_checked",
                           "p1_min_raise", "p1_max_raise",
                           "check_min_raise", "check_max_raise"]:
            hist_atom = getattr(d.history, hist_name)
            chunk_name = f"{strength_name}_{hist_name}"
            chunk = chunk_name ^ + wm ** card_atom + wm ** hist_atom
            defs.append(chunk)
    return defs


RULE_RECOMMENDATIONS = {
    "strong_start":           "max_raise",
    "strong_p1_checked":      "max_raise",
    "strong_p1_min_raise":    "call",
    "strong_p1_max_raise":    "call",
    "strong_check_min_raise": "call",
    "strong_check_max_raise": "call",
    "medium_start":           "check",
    "medium_p1_checked":      "check",
    "medium_p1_min_raise":    "fold",
    "medium_p1_max_raise":    "fold",
    "medium_check_min_raise": "fold",
    "medium_check_max_raise": "fold",
    "weak_start":             "check",
    "weak_p1_checked":        "check",
    "weak_p1_min_raise":      "fold",
    "weak_p1_max_raise":      "fold",
    "weak_check_min_raise":   "fold",
    "weak_check_max_raise":   "fold",
}
