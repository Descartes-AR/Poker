"""MS — single profit drive. r = drive * chip_delta (identity scaling in v2).

The computed reward is written BOTH to the MS's own reward_out site (so the
MCS can read it for its prediction-error bookkeeping) AND directly onto the
selector's reward site (so the TD update can consume it). The selector's
reward site is passed in at construction as `reward_target`.
"""

from datetime import timedelta
from pyClarion import Priority, Event
from pyClarion.events import ForwardUpdate, State, Site
from pyClarion.components.base import Component

from keyspace import PokerKeyspace


class MSModule(Component):
    chip_in:    Site = Site()
    drive_state: Site = Site()
    reward_out: Site = Site()

    def __init__(self, name, root: PokerKeyspace, reward_target: State | None = None):
        super().__init__(name)
        self.root = root
        # The TDLearning.reward site to write into. Stored as a plain
        # attribute (NOT a declared Site) so pyClarion doesn't treat it as
        # one of this component's own sites.
        self._reward_target = reward_target
        d, b = root.d, root.b
        idx_rwd, idx_drive = self._init_indexes(
            (b.main.rwd, d.reward), (b.main.drive, d.drive))
        self.chip_in     = State(idx_rwd, {}, 0.0)
        self.drive_state = State(idx_drive,
            {~b.main.drive * ~d.drive.profit: 1.0}, 0.0)
        self.reward_out  = State(idx_rwd, {}, 0.0)

    def resolve(self, event):
        if self.chip_in in event.index(ForwardUpdate):
            self.system.schedule(self.compute_reward())

    def compute_reward(self, dt=timedelta(), priority=Priority.PROPAGATION):
        d, b = self.root.d, self.root.b
        profit = self.drive_state[0][~b.main.drive * ~d.drive.profit]
        chip_delta = self.chip_in[0][~b.main.rwd * ~d.reward.main]
        r = profit * chip_delta
        reward_key = ~b.main.rwd * ~d.reward.main
        updates = [ForwardUpdate(self.reward_out, {reward_key: r}, "write")]
        # Write the reward directly onto the selector's reward site so the
        # TD update can consume it on the next decision.
        if self._reward_target is not None:
            updates.append(ForwardUpdate(self._reward_target, {reward_key: r}, "write"))
        return Event(self.compute_reward, updates, dt, priority)
