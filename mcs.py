"""MCS — adaptive Pool weights based on per-level prediction accuracy.
Action space size doesn't affect the algorithm — we read whichever Q-value
corresponds to the chosen action, regardless of total action count."""

from datetime import timedelta
from pyClarion import Priority, Event
from pyClarion.events import ForwardUpdate, State, Site
from pyClarion.components.base import Component

from keyspace import PokerKeyspace


class MCSModule(Component):
    last_chosen: Site = Site()
    last_q_top:  Site = Site()
    last_q_bot:  Site = Site()

    def __init__(self, name, root, acs_module, ms_module,
                 *, eta=1e-3, sum_w=2.0):
        super().__init__(name)
        self.root = root
        self.acs = acs_module
        self.ms = ms_module
        self.eta = eta
        self.sum_w = sum_w
        d, b = root.d, root.b
        action_d = (b.main.wm, d.action)
        (idx_action,) = self._init_indexes(action_d)
        self.last_chosen = State(idx_action, {}, 0.0)
        self.last_q_top  = State(idx_action, {}, 0.0)
        self.last_q_bot  = State(idx_action, {}, 0.0)
        # Pool weight key caching — populated after construction
        self._w_top_key = None
        self._w_bot_key = None

    def resolve(self, event):
        if event.source == self.acs.selector.select:
            self.system.schedule(self.snapshot())
        if event.source == self.ms.compute_reward:
            self.system.schedule(self.adapt_weights())

    def snapshot(self, dt=timedelta(), priority=Priority.LEARNING):
        return Event(self.snapshot, [
            ForwardUpdate(self.last_chosen, dict(self.acs.selector.main[0].d), "write"),
            ForwardUpdate(self.last_q_top,  dict(self.acs.rules.main[0].d), "write"),
            ForwardUpdate(self.last_q_bot,  dict(self.acs.bottom.main[0].d), "write"),
        ], dt, priority)

    def adapt_weights(self, dt=timedelta(), priority=Priority.LEARNING):
        d, b = self.root.d, self.root.b
        r = self.ms.reward_out[0][~b.main.rwd * ~d.reward.main]
        chosen = self.last_chosen[0].d
        if not chosen:
            return Event(self.adapt_weights, [], dt, priority)
        chosen_key = max(chosen, key=chosen.get)
        q_top_chosen = self.last_q_top[0].d.get(chosen_key, 0.0)
        q_bot_chosen = self.last_q_bot[0].d.get(chosen_key, 0.0)
        d_top = (r - q_top_chosen) ** 2
        d_bot = (r - q_bot_chosen) ** 2
        diff = self.eta * (d_top - d_bot)   # >0 if top is worse → shift to bottom

        # On first call, identify the Pool weight keys
        pool_params = dict(self.acs.pool.params[0].d)
        if self._w_top_key is None:
            # The Pool's per-input weight keys are named after each input
            # component. After init, two atoms exist in self.acs.pool.p:
            # one named after self.acs.bottom, one after self.acs.rules.
            for key, _ in pool_params.items():
                key_str = str(key)
                if "rules" in key_str:  self._w_top_key = key
                if "bottom" in key_str: self._w_bot_key = key

        if self._w_top_key is None or self._w_bot_key is None:
            return Event(self.adapt_weights, [], dt, priority)

        w_top = pool_params[self._w_top_key]
        w_bot = pool_params[self._w_bot_key]
        w_top_new = w_top - diff
        w_bot_new = w_bot + diff
        # Normalize to sum=self.sum_w
        total = w_top_new + w_bot_new
        if total > 0:
            w_top_new *= self.sum_w / total
            w_bot_new *= self.sum_w / total
        return Event(self.adapt_weights,
            [ForwardUpdate(self.acs.pool.params,
                {self._w_top_key: w_top_new, self._w_bot_key: w_bot_new}, "write")],
            dt, priority)
