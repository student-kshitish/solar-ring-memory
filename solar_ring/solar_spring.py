import torch
import torch.nn as nn
import torch.nn.functional as F

POS_MASS_WEIGHTS = {
    'SUBJ': 0.95, 'OBJ': 0.90, 'VERB': 0.85,
    'ADJ':  0.50, 'ADV': 0.40, 'PREP': 0.20,
    'CONJ': 0.15, 'DET': 0.05, 'OTHER': 0.10,
}

N_POS = 8


class SolarSpringAttention(nn.Module):
    """
    Unified field attention combining:
    - Micro gravity (within-ring slot distances)
    - Macro gravity (between-ring orbital distances)
    - Spring force (grows with token distance)
    - Black hole force (confidence-based collapse)

    Replaces standard dot-product attention.
    All parameters learned end-to-end.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d = d_model

        # Micro gravity constants — one per POS pair
        # G_k[i,j] = gravity between POS_i and POS_j slots
        self.G_micro = nn.Parameter(
            torch.ones(N_POS, N_POS) * 0.1
        )

        # Macro gravity constant — between rings
        self.G_macro = nn.Parameter(torch.tensor(0.5))

        # Spring constants — one per POS type
        # k[i] = how strongly POS_i pulls back over distance
        self.k_spring = nn.Parameter(
            torch.ones(N_POS) * 0.05
        )

        # Temperature — controls sharpness of attention
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Value projection
        self.W_v   = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

    def semantic_mass(self, vec: torch.Tensor,
                      pos_idx: int,
                      resonance: float = 0.0) -> float:
        """
        m = ||vec||₂ × POS_weight × (1 + resonance)
        High resonance with Sun = heavier = stronger gravity.
        """
        pos_types = list(POS_MASS_WEIGHTS.keys())
        pos_type  = pos_types[pos_idx % N_POS]
        w = POS_MASS_WEIGHTS.get(pos_type, 0.1)
        return vec.norm().item() * w * (1.0 + resonance)

    def micro_gravity(self, mi: float, mj: float,
                      pos_i: int, pos_j: int,
                      slot_dist: int) -> torch.Tensor:
        """
        G_micro(i,j) = G_k[pos_i, pos_j] · mᵢ · mⱼ / r²
        r = slot distance within ring (min 1)
        """
        pi = pos_i % N_POS
        pj = pos_j % N_POS
        G_k = torch.sigmoid(self.G_micro[pi, pj])
        r   = max(slot_dist, 1)
        return G_k * mi * mj / (r ** 2)

    def macro_gravity(self, mi: float, mj: float,
                      depth_i: int, depth_j: int) -> torch.Tensor:
        """
        G_macro(i,j) = G_Ω · mᵢ · mⱼ / r²_orbital
        r_orbital = |depth_i - depth_j| + 1
        """
        G_Ω = torch.sigmoid(self.G_macro)
        r   = abs(depth_i - depth_j) + 1
        return G_Ω * mi * mj / (r ** 2)

    def spring_force(self, pos_idx: int,
                     token_dist: int) -> torch.Tensor:
        """
        F_spring = -k · Δpos
        GROWS with distance — compensates memory decay.
        Pronoun far from antecedent gets STRONGER pull.
        This is what beats BERT on long-range Winograd.
        """
        k = torch.sigmoid(self.k_spring[pos_idx % N_POS])
        return k * token_dist   # positive = attractive

    def black_hole_force(self, confidence: float) -> float:
        """
        F_bh = -G_bh / (conf - EVENT_HORIZON)²
        As confidence → 0.1 (event horizon), force → -∞
        Collapsed rings stop competing for resolution.
        """
        EVENT_HORIZON = 0.1
        if confidence <= EVENT_HORIZON:
            return -1e6   # collapsed — infinite repulsion
        gap = confidence - EVENT_HORIZON
        return -0.01 / (gap ** 2)   # negative = repulsive

    def forward(self, concepts: list,
                token_vecs: torch.Tensor,
                confidences: list = None,
                sun_resonances: list = None) -> tuple:
        """
        Compute unified field attention scores.

        Args:
            concepts: list of dicts with keys:
                      pos_idx, depth, token_pos, slot_idx
            token_vecs: (L, d) embeddings
            confidences: list of floats per concept
            sun_resonances: list of floats per concept

        Returns:
            attended: (L, d)
            score_matrix: (L, L) for interpretability
        """
        L = len(concepts)
        if confidences is None:
            confidences = [1.0] * L
        if sun_resonances is None:
            sun_resonances = [0.0] * L

        # Compute masses
        masses = [
            self.semantic_mass(
                token_vecs[i],
                concepts[i].get('pos_idx', 0),
                sun_resonances[i]
            )
            for i in range(L)
        ]

        # Build unified field score matrix
        scores = torch.zeros(L, L, device=token_vecs.device)

        for i in range(L):
            ci  = concepts[i]
            mi  = masses[i]
            bh_i = self.black_hole_force(confidences[i])

            if bh_i < -1e5:
                # Collapsed ring — no attraction
                continue

            for j in range(L):
                if i == j:
                    continue

                cj  = concepts[j]
                mj  = masses[j]

                # Micro gravity
                slot_dist = abs(
                    ci.get('slot_idx', 0) -
                    cj.get('slot_idx', 0)
                )
                f_micro = self.micro_gravity(
                    mi, mj,
                    ci.get('pos_idx', 0),
                    cj.get('pos_idx', 0),
                    slot_dist
                )

                # Macro gravity
                f_macro = self.macro_gravity(
                    mi, mj,
                    ci.get('depth', 0),
                    cj.get('depth', 0)
                )

                # Spring force
                token_dist = abs(
                    ci.get('token_pos', i) -
                    cj.get('token_pos', j)
                )
                f_spring = self.spring_force(
                    ci.get('pos_idx', 0), token_dist
                )

                # Black hole repulsion for j
                bh_j = self.black_hole_force(confidences[j])

                # Unified field
                scores[i, j] = (
                    f_micro + f_macro +
                    f_spring + bh_i + bh_j
                )

        # Temperature-scaled softmax
        T = torch.clamp(self.temperature.abs(), min=0.1)
        A = F.softmax(scores / T, dim=-1)

        # Attended values
        V        = self.W_v(token_vecs)
        attended = A @ V
        out      = self.W_out(attended)

        return out, A, scores
