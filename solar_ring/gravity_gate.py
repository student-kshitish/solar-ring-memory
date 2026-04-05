"""GravityGate: controls which tokens persist in ring memory by POS mass."""

import torch
import torch.nn as nn
from .sun_state import SunState

POS_MASS = {
    'SUBJ':  0.95,   # nouns as subject — highest mass
    'OBJ':   0.90,   # nouns as object — high mass
    'VERB':  0.85,   # verbs — high mass
    'ADJ':   0.50,   # adjectives — medium mass
    'ADV':   0.40,   # adverbs — medium mass
    'PREP':  0.20,   # prepositions — low mass
    'CONJ':  0.15,   # conjunctions — low mass
    'DET':   0.05,   # determiners — lowest mass (ejected fast)
    'OTHER': 0.10,   # other — very low mass
}


class GravityGate(nn.Module):
    """
    Controls which tokens stay in ring memory.
    High mass tokens (nouns/verbs) persist longer.
    Low mass tokens (articles/fillers) decay quickly.

    Gate = σ(W_pos · x_t + b) × pos_mass × resonance_boost
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.W_gate = nn.Linear(d_model, 1)
        self.pos_mass = POS_MASS

    def forward(
        self,
        token_vec: torch.Tensor,
        pos_type: str,
        sun_state: SunState = None,
    ) -> float:
        """
        Returns gate value 0.0 to 1.0.
        High gate = keep this token in ring.
        Low gate  = let this token decay quickly.
        """
        base_gate = torch.sigmoid(self.W_gate(token_vec.float()))
        mass = self.pos_mass.get(pos_type, 0.10)

        resonance_boost = 1.0
        if sun_state is not None:
            res = sun_state.resonance(token_vec)
            if res > 0.5:
                resonance_boost = 2.0  # double pull for high resonance

        gate = base_gate.item() * mass * resonance_boost
        return min(gate, 1.0)
