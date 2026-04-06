import torch
import torch.nn as nn
import torch.nn.functional as F

POS_MASS_WEIGHTS = {
    'SUBJ': 0.95, 'OBJ': 0.90, 'VERB': 0.85,
    'ADJ':  0.50, 'ADV': 0.40, 'PREP': 0.20,
    'CONJ': 0.15, 'DET': 0.05, 'OTHER': 0.10,
}

N_POS = 8

# Isotope decay constants — how fast each POS type fades from memory
DECAY_CONSTANTS = {
    0: 0.005,   # SUBJ — nearly permanent
    1: 0.030,   # VERB — fades after clause
    2: 0.005,   # OBJ  — nearly permanent
    3: 0.080,   # PREP — local
    4: 0.050,   # CONJ — medium
    5: 0.100,   # ADJ  — local modifier
    6: 0.100,   # ADV  — local modifier
    7: 0.350,   # DET  — immediately ejected
}


class SolarSpringAttention(nn.Module):
    """
    Unified field attention combining:
    - Micro gravity (within-ring slot distances)
    - Macro gravity (between-ring orbital distances)
    - Spring force (grows with token distance)
    - Black hole force (confidence-based collapse)
    - Neutron star force (compressed collapsed rings)
    - Centripetal/centrifugal orbital balance
    - Lagrange point distance prioritization
    - Isotope decay confidences per POS type

    Replaces standard dot-product attention.
    All parameters learned end-to-end.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d = d_model

        # Micro gravity constants — one per POS pair
        self.G_micro = nn.Parameter(
            torch.ones(N_POS, N_POS) * 0.1
        )

        # Macro gravity constant — between rings
        self.G_macro = nn.Parameter(torch.tensor(0.5))

        # Spring constants — one per POS type
        self.k_spring = nn.Parameter(
            torch.ones(N_POS) * 0.05
        )

        # Temperature — controls sharpness of attention
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Learned decay constants per POS (initialized from isotope model)
        decay_init = torch.tensor(
            [DECAY_CONSTANTS[i] for i in range(N_POS)],
            dtype=torch.float
        )
        self.lambda_decay = nn.Parameter(decay_init)

        # Neutron star compression factor
        self.G_neutron = nn.Parameter(torch.tensor(2.0))

        # Centripetal/centrifugal balance
        self.omega = nn.Parameter(torch.tensor(0.1))

        # Lagrange point learnable
        self.r_star = nn.Parameter(torch.tensor(3.0))

        # Value projection
        self.W_v   = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

    # ── Legacy scalar helpers (kept for backward compatibility) ────────

    def semantic_mass(self, vec: torch.Tensor,
                      pos_idx: int,
                      resonance: float = 0.0) -> float:
        pos_types = list(POS_MASS_WEIGHTS.keys())
        pos_type  = pos_types[pos_idx % N_POS]
        w = POS_MASS_WEIGHTS.get(pos_type, 0.1)
        return vec.norm().item() * w * (1.0 + resonance)

    def micro_gravity(self, mi: float, mj: float,
                      pos_i: int, pos_j: int,
                      slot_dist: int) -> torch.Tensor:
        pi = pos_i % N_POS
        pj = pos_j % N_POS
        G_k = torch.sigmoid(self.G_micro[pi, pj])
        r   = max(slot_dist, 1)
        return G_k * mi * mj / (r ** 2)

    def macro_gravity(self, mi: float, mj: float,
                      depth_i: int, depth_j: int) -> torch.Tensor:
        G_Ω = torch.sigmoid(self.G_macro)
        r   = abs(depth_i - depth_j) + 1
        return G_Ω * mi * mj / (r ** 2)

    def spring_force(self, pos_idx: int,
                     token_dist: int) -> torch.Tensor:
        k = torch.sigmoid(self.k_spring[pos_idx % N_POS])
        return k * token_dist

    def black_hole_force(self, confidence: float) -> float:
        EVENT_HORIZON = 0.1
        if confidence <= EVENT_HORIZON:
            return -1e6
        gap = confidence - EVENT_HORIZON
        return -0.01 / (gap ** 2)

    # ── New physics methods ────────────────────────────────────────────

    def compute_decay_confidences(self, pos_idx_tensor: torch.Tensor,
                                   token_pos_tensor: torch.Tensor,
                                   L: int) -> torch.Tensor:
        """
        conf(i) = e^(-λ_pos × age)
        age = max_token_pos - token_pos_i
        """
        lambdas = torch.abs(self.lambda_decay)
        lam_i   = lambdas[pos_idx_tensor % N_POS]    # (L,)
        max_pos = token_pos_tensor.max()
        age     = max_pos - token_pos_tensor           # (L,)
        confs   = torch.exp(-lam_i * age.float())      # (L,)
        return confs

    def neutron_star_force(self, confs: torch.Tensor,
                            mi: torch.Tensor, mj: torch.Tensor,
                            r: torch.Tensor) -> torch.Tensor:
        """
        Collapsed rings (conf < 0.15) become neutron stars.
        Ultra-compressed but still exert gravity.
        F_ns = G_ns * compression_i * compression_j * mi * mj / r^2
        """
        G_ns          = torch.abs(self.G_neutron)
        compression_i = 1.0 / torch.clamp(confs.unsqueeze(1), min=0.01)
        compression_j = 1.0 / torch.clamp(confs.unsqueeze(0), min=0.01)
        collapsed_i   = (confs < 0.15).float().unsqueeze(1)
        collapsed_j   = (confs < 0.15).float().unsqueeze(0)
        F_ns = (G_ns * compression_i * compression_j *
                mi * mj / torch.clamp(r ** 2, min=0.1))
        return F_ns * collapsed_i * collapsed_j

    def centripetal_centrifugal(self, masses: torch.Tensor,
                                 slot_dist: torch.Tensor) -> torch.Tensor:
        """
        Net orbital force = centripetal - centrifugal
        F_cp = mi * mj / r   (inward)
        F_cf = mi * omega^2 * r  (outward)
        Equilibrium at r* = m/omega (Lagrange point)
        """
        omega = torch.abs(self.omega)
        mi    = masses.unsqueeze(1)
        mj    = masses.unsqueeze(0)
        r     = torch.clamp(slot_dist.float(), min=0.1)
        F_centripetal = mi * mj / r
        F_centrifugal = mi * omega ** 2 * r
        return F_centripetal - F_centrifugal

    def lagrange_boost(self, token_dist: torch.Tensor) -> torch.Tensor:
        """
        Maximum attraction at Lagrange distance r*.
        boost = 1 - |r - r*| / r*
        Peaked at r = r*, decays away.
        """
        r_star    = torch.abs(self.r_star)
        deviation = (token_dist.float() - r_star).abs()
        boost     = 1.0 - deviation / (r_star + 1.0)
        return torch.clamp(boost, min=0.0)

    # ── Vectorized forward ─────────────────────────────────────────────

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
            confidences:    ignored — computed from isotope decay
            sun_resonances: list of floats per concept (optional)

        Returns:
            out:    (L, d)
            A:      (L, L) attention weights
            scores: (L, L) raw unified field scores
        """
        L = len(concepts)
        if L == 0:
            return token_vecs, None, None

        device = token_vecs.device

        if sun_resonances is None:
            sun_resonances = [0.0] * L

        # ── Extract concept tensors ────────────────────────────────────
        pos_idx   = torch.tensor(
            [c.get('pos_idx', 0) % N_POS for c in concepts],
            device=device, dtype=torch.long)
        depths    = torch.tensor(
            [float(c.get('depth', 0)) for c in concepts],
            device=device)
        token_pos = torch.tensor(
            [float(c.get('token_pos', i)) for i, c in enumerate(concepts)],
            device=device)
        slot_idx  = torch.tensor(
            [float(c.get('slot_idx', 0)) for c in concepts],
            device=device)
        res_t     = torch.tensor(sun_resonances, device=device,
                                 dtype=torch.float)

        # ── Isotope decay confidences (replaces fixed confidences) ─────
        confs = self.compute_decay_confidences(pos_idx, token_pos, L)

        # ── Semantic masses ────────────────────────────────────────────
        pos_types = list(POS_MASS_WEIGHTS.keys())
        pw_list   = [POS_MASS_WEIGHTS.get(pos_types[p % N_POS], 0.1)
                     for p in range(N_POS)]
        pos_w_t   = torch.tensor(pw_list, device=device)
        w         = pos_w_t[pos_idx]                        # (L,)
        norms     = token_vecs.norm(dim=-1).float()         # (L,)
        masses    = norms * w * (1.0 + res_t)               # (L,)

        # ── Pairwise distance matrices (L, L) ─────────────────────────
        slot_dist  = (slot_idx.unsqueeze(1) -
                      slot_idx.unsqueeze(0)).abs()
        depth_dist = (depths.unsqueeze(1) -
                      depths.unsqueeze(0)).abs()
        token_dist = (token_pos.unsqueeze(1) -
                      token_pos.unsqueeze(0)).abs()

        mi = masses.unsqueeze(1)    # (L, 1)
        mj = masses.unsqueeze(0)    # (1, L)

        # ── Micro gravity ──────────────────────────────────────────────
        pi    = pos_idx.unsqueeze(1).expand(L, L)
        pj    = pos_idx.unsqueeze(0).expand(L, L)
        G_k   = torch.sigmoid(self.G_micro)[pi, pj]
        r_slot = slot_dist.clamp(min=1).float()
        F_micro = G_k * mi * mj / r_slot.pow(2)

        # ── Macro gravity ──────────────────────────────────────────────
        G_Ω    = torch.sigmoid(self.G_macro)
        r_orb  = (depth_dist + 1).float()
        F_macro = G_Ω * mi * mj / r_orb.pow(2)

        # ── Spring force ───────────────────────────────────────────────
        k_pi    = torch.sigmoid(self.k_spring)[pos_idx]     # (L,)
        F_spring = k_pi.unsqueeze(1) * token_dist           # (L, L)

        # ── Neutron star force ─────────────────────────────────────────
        F_ns = self.neutron_star_force(confs, mi, mj, r_slot)

        # ── Centripetal / centrifugal orbital balance ──────────────────
        F_orbital = self.centripetal_centrifugal(masses, slot_dist)

        # ── Lagrange distance boost ────────────────────────────────────
        F_lagrange = self.lagrange_boost(token_dist)

        # ── Complete unified field ─────────────────────────────────────
        scores = (F_micro + F_macro + F_spring +
                  F_ns + F_orbital + F_lagrange)

        # Confidence-weighted softmax
        conf_weight = confs.unsqueeze(1) * confs.unsqueeze(0)
        scores = scores * conf_weight
        scores.fill_diagonal_(0)

        T = torch.clamp(self.temperature.abs(), min=0.1)
        A = torch.softmax(scores / T, dim=-1)

        V        = self.W_v(token_vecs.float())
        attended = A @ V
        out      = self.W_out(attended)

        return out, A, scores
