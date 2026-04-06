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
        Compute unified field attention scores (fully vectorized).

        Args:
            concepts: list of dicts with keys:
                      pos_idx, depth, token_pos, slot_idx
            token_vecs: (L, d) embeddings
            confidences: list of floats per concept
            sun_resonances: list of floats per concept

        Returns:
            out: (L, d)
            A:  (L, L) attention weights
            scores: (L, L) raw unified field scores
        """
        L   = len(concepts)
        dev = token_vecs.device
        if confidences is None:
            confidences = [1.0] * L
        if sun_resonances is None:
            sun_resonances = [0.0] * L

        # ── Extract concept arrays ───────────────────────────────
        pos_idx_t  = torch.tensor(
            [c.get('pos_idx', 0) % N_POS for c in concepts],
            device=dev, dtype=torch.long)
        depth_t    = torch.tensor(
            [float(c.get('depth', 0)) for c in concepts],
            device=dev)
        tok_pos_t  = torch.tensor(
            [float(c.get('token_pos', i))
             for i, c in enumerate(concepts)],
            device=dev)
        slot_idx_t = torch.tensor(
            [float(c.get('slot_idx', 0)) for c in concepts],
            device=dev)
        conf_t     = torch.tensor(confidences, device=dev)
        res_t      = torch.tensor(sun_resonances, device=dev)

        # ── Semantic masses ──────────────────────────────────────
        pos_types  = list(POS_MASS_WEIGHTS.keys())
        pw_list    = [POS_MASS_WEIGHTS.get(pos_types[p % N_POS], 0.1)
                      for p in range(N_POS)]
        pos_w_t    = torch.tensor(pw_list, device=dev)
        w          = pos_w_t[pos_idx_t]                 # (L,)
        norms      = token_vecs.norm(dim=-1)             # (L,)
        masses     = norms * w * (1.0 + res_t)           # (L,)

        # ── Black hole forces  ───────────────────────────────────
        EVENT_HORIZON = 0.1
        gap    = (conf_t - EVENT_HORIZON).clamp(min=1e-6)
        bh     = -0.01 / gap.pow(2)
        bh     = torch.where(conf_t <= EVENT_HORIZON,
                             torch.full_like(bh, -1e6), bh)

        # ── Pairwise (L, L) fields ───────────────────────────────
        mi = masses.unsqueeze(1)          # (L, 1)
        mj = masses.unsqueeze(0)          # (1, L)

        # Micro gravity: sigmoid(G_micro[pi,pj]) * mi*mj / dist²
        pi    = pos_idx_t.unsqueeze(1).expand(L, L)
        pj    = pos_idx_t.unsqueeze(0).expand(L, L)
        G_k   = torch.sigmoid(self.G_micro)[pi, pj]     # (L, L)
        sd    = (slot_idx_t.unsqueeze(1) -
                 slot_idx_t.unsqueeze(0)).abs().clamp(min=1)
        f_micro = G_k * mi * mj / sd.pow(2)

        # Macro gravity: sigmoid(G_macro) * mi*mj / depth_dist²
        G_Ω   = torch.sigmoid(self.G_macro)
        dd    = (depth_t.unsqueeze(1) -
                 depth_t.unsqueeze(0)).abs() + 1
        f_macro = G_Ω * mi * mj / dd.pow(2)

        # Spring force: sigmoid(k[pi]) * token_dist
        k_pi  = torch.sigmoid(self.k_spring)[pos_idx_t]  # (L,)
        td    = (tok_pos_t.unsqueeze(1) -
                 tok_pos_t.unsqueeze(0)).abs()
        f_spring = k_pi.unsqueeze(1) * td               # (L, L)

        # Black hole (per row and col)
        bh_i = bh.unsqueeze(1).expand(L, L)
        bh_j = bh.unsqueeze(0).expand(L, L)

        # Collapsed rows → no attraction
        collapsed = (conf_t <= EVENT_HORIZON)
        row_mask  = collapsed.unsqueeze(1).expand(L, L)

        # Unified field matrix
        scores = f_micro + f_macro + f_spring + bh_i + bh_j
        scores = scores.masked_fill(row_mask, -1e6)

        # Zero diagonal
        eye    = torch.eye(L, device=dev, dtype=torch.bool)
        scores = scores.masked_fill(eye, 0.0)

        # Temperature-scaled softmax
        T = torch.clamp(self.temperature.abs(), min=0.1)
        A = F.softmax(scores / T, dim=-1)

        # Attended values
        V        = self.W_v(token_vecs)
        attended = A @ V
        out      = self.W_out(attended)

        return out, A, scores
