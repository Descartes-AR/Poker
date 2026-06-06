"""NACS — unchanged from the simpler version. Card-strength concepts."""

from pyClarion import ChunkStore
from pyClarion.components.base import Component

from keyspace import PokerKeyspace


class NACSModule(Component):
    def __init__(self, name, root: PokerKeyspace):
        super().__init__(name)
        self.root = root
        b, d, k = root.b, root.d, root.k_nacs
        feat_d = (b.main.wm, d)
        with self:
            self.chunks = ChunkStore(f"{name}.chunks", c=k, d=feat_d)
            self.bu = self.chunks.bottom_up(f"{name}.bu")


def nacs_chunk_defs(root):
    d = root.d
    wm = root.b.main.wm
    return [
        "strong_hand" ^ + wm ** d.card.k,
        "medium_hand" ^ + wm ** d.card.q,
        "weak_hand"   ^ + wm ** d.card.j,
    ]
