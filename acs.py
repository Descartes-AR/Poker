"""
ACS module — TWO-PATH design (learning bypasses the cam Pool).

This mirrors the working Lab 6 pattern on the learning side, which is the key
fix: the trainable bottom layer feeds the TDLearning selector DIRECTLY, so the
TD gradient flows back through a clean differentiable Layer (cam is never in the
gradient path). The earlier design put a cam Pool between bottom and selector;
because cam has no gradient (cam.grad raises NotImplementedError), no gradient
ever reached the bottom weights and nothing learned.

  LEARNING PATH (differentiable, identical in shape to Lab 6):
      ipt -> bottom(Layer) -> selector(TDLearning)
      The selector selects over the bottom layer's Q-values, computes the TD
      error, and backpropagates it through `bottom` only. Adam then steps.

  SELECTION / BEHAVIOUR PATH (forward-only, the cam-override mechanism):
      (bottom, rules) -> pool(cam)
      The Pool combines the learned bottom Q-values with the fixed rule
      recommendations via cam. The ActionSensor (simulation.py) reads the Pool
      to decide the action actually PLAYED — this is where a strong learned
      bottom signal can override an explicit rule (the bluffing story).

  ATTRIBUTION ALIGNMENT:
      Because behaviour follows the cam Pool but learning runs on the selector,
      the selector must learn the value of the action that was actually played.
      The ActionSensor overwrites the selector's recorded action with the
      cam-chosen action (see simulation.py). This makes the TD update target
      Q(s, a_played), which is correct off-policy Q-learning.
"""

from pyClarion import Layer, Pool, ChunkStore
from pyClarion.components.learning import TDLearning
from pyClarion.components.optimizers import Adam
from pyClarion.components.base import Component
from pyClarion.events import BackwardUpdate, ForwardUpdate

from keyspace import PokerKeyspace


# ── Python 3.14 compatibility shim ───────────────────────────────────────
# The gradient tape calls inspect.signature() on ops. On 3.14 the `cam`
# INSTANCE has no derivable signature (the CAM class does). We attach an
# explicit __signature__ so any tape recording succeeds. (cam is now only in
# the forward-only selection path, so this rarely matters, but it is harmless
# and future-proofs any differentiable use.)
def _patch_cam_signature():
    import inspect
    from pyClarion.components import ops as _ops
    cam_instance = getattr(_ops, "cam", None)
    CAM_cls = getattr(_ops, "CAM", None)
    if cam_instance is None or CAM_cls is None:
        return
    try:
        inspect.signature(cam_instance)
        return
    except (ValueError, TypeError):
        pass
    try:
        call_sig = inspect.signature(CAM_cls.__call__)
        params = [p for name, p in call_sig.parameters.items() if name != "self"]
        cam_instance.__signature__ = call_sig.replace(parameters=params)
    except Exception:
        pass


_patch_cam_signature()


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

            # ── LEARNING PATH: bottom -> selector (Lab 6 pattern) ──
            # selector.input is bottom.main, so TD backprop flows through the
            # bottom layer ONLY. This is the differentiable learning path.
            self.selector = self.bottom >> TDLearning(
                f"{name}.selector",
                p=p, s=s, d=action_d, r=reward_d,
                gamma=gamma, sd=sd, f=0.0, l=2)

            # ── SELECTION PATH: (bottom, rules) -> cam Pool (forward only) ──
            # The Pool combines learned bottom Q with fixed rule recs for the
            # behaviour decision. It is never differentiated, so cam's lack of
            # a gradient is irrelevant here.
            self.pool = Pool(f"{name}.pool", p, action_d, l=2)
            (self.bottom, self.rules) >> self.pool

            # ── Optimizer (BOTTOM only) ──
            self.optimizer = Adam(f"{name}.optimizer", p, lr=lr)
            self.optimizer.add(self.bottom.weights, self.bottom.bias)

    def resolve(self, event):
        forward = event.index(ForwardUpdate)
        # Drive the cam Pool's forward pass once per decision, after the rules
        # layer finishes (the later-updating of the Pool's two inputs, so both
        # bottom.main and rules.main are fresh). This produces the
        # cam(bottom_Q, rule_recs) the ActionSensor reads for behaviour.
        if self.rules.main in forward:
            self.system.schedule(self.pool.forward())
        # After the bottom layer's backward pass accumulates gradients, trigger
        # the optimizer to consume and apply them (Adam.resolve is a no-op).
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
