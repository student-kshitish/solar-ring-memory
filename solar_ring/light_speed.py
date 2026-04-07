import torch
import torch.nn as nn
import math

# Speed of light constant in memory space
# Learned parameter — different domains have different c
DEFAULT_C = 50.0  # tokens

# Particle masses per POS (0=massless, 1=maximum mass)
PARTICLE_MASS = {
    'SUBJ':  0.95,   # noun as subject — very massive
    'OBJ':   0.90,   # noun as object — very massive
    'VERB':  0.85,   # verb — massive
    'ADJ':   0.50,   # adjective — medium
    'ADV':   0.40,   # adverb — medium
    'PREP':  0.20,   # preposition — light
    'CONJ':  0.15,   # conjunction — very light
    'DET':   0.05,   # determiner — nearly massless
    'PRON':  0.00,   # pronoun — MASSLESS (photon)
    'OTHER': 0.10,   # other — very light
}

POS_NAMES = list(PARTICLE_MASS.keys())


class LightSpeedMemory(nn.Module):
    """
    Speed-of-light constrained memory.

    Key mechanisms:
    1. Causal light cone — only past tokens within cone
       can influence current token resolution
    2. Redshift — information fades with distance
       λ(d) = e^(-d/c_memory)
    3. Photon particles — pronouns massless, travel at c
    4. Massive particles — nouns/verbs travel at v < c

    This enforces causal consistency missing from transformers.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d = d_model

        # Learnable speed of light
        self.log_c = nn.Parameter(
            torch.tensor(math.log(DEFAULT_C))
        )

        # Learnable particle masses per POS
        mass_init = torch.tensor(
            [PARTICLE_MASS[p] for p in POS_NAMES],
            dtype=torch.float
        )
        self.particle_mass = nn.Parameter(mass_init)

        # Redshift projection
        self.W_redshift = nn.Linear(d_model, d_model)

        # Causal gate
        self.W_causal = nn.Linear(d_model * 2, 1)

    @property
    def c_memory(self) -> float:
        """Speed of light in memory space (token units)."""
        return torch.exp(self.log_c).item()

    def get_mass(self, pos_idx: int) -> float:
        """Get particle mass for POS type."""
        idx = pos_idx % len(POS_NAMES)
        return torch.sigmoid(self.particle_mass[idx]).item()

    def particle_velocity(self, pos_idx: int) -> float:
        """
        v = (1 - mass) × c_memory
        Massless (pronoun): v = c_memory (full speed)
        Massive (noun): v = 0.05 × c_memory (slow)
        """
        mass = self.get_mass(pos_idx)
        return (1.0 - mass) * self.c_memory

    def redshift(self, distance: float,
                 pos_idx: int = 0) -> float:
        """
        λ(d) = e^(-d/c_memory)
        Massless particles (photons): no redshift
        Massive particles: exponential fade
        """
        mass = self.get_mass(pos_idx)
        if mass < 0.01:
            return 1.0  # photon — no redshift

        c = self.c_memory
        return math.exp(-distance / c)

    def causal_mask(self,
                    pos_i: int,
                    pos_j: int,
                    pos_idx_j: int) -> float:
        """
        Can token at pos_j influence token at pos_i?

        Condition: pos_j < pos_i (past only)
                   AND pos_j within light cone of pos_i

        Light cone boundary:
            pos_i - pos_j <= c_memory / velocity_j

        Massless (photon): can influence from anywhere in past
        Massive: limited by their slower velocity
        """
        if pos_j >= pos_i:
            return 0.0  # future cannot influence past

        distance = pos_i - pos_j
        v = self.particle_velocity(pos_idx_j)

        if v < 1e-6:
            return 0.0  # stationary — no influence

        # Time for information to travel from j to i
        travel_time = distance / v

        # If travel time > 1 (one token step) it's outside cone
        if travel_time > self.c_memory:
            return 0.0  # outside light cone

        return 1.0

    def compute_light_speed_scores(self,
                                   token_vecs: torch.Tensor,
                                   token_positions: list,
                                   pos_indices: list,
                                   base_scores: torch.Tensor
                                   ) -> torch.Tensor:
        """
        Apply light-speed constraints to attention scores.

        Args:
            token_vecs: (L, d)
            token_positions: list of int token positions
            pos_indices: list of int POS indices
            base_scores: (L, L) base gravity scores

        Returns:
            constrained_scores: (L, L)
        """
        L = len(token_positions)
        device = token_vecs.device

        # Build redshift matrix (L, L)
        redshift_mat = torch.zeros(L, L, device=device)
        causal_mat   = torch.zeros(L, L, device=device)

        for i in range(L):
            for j in range(L):
                if i == j:
                    continue

                d = abs(token_positions[i] - token_positions[j])

                # Redshift
                rs = self.redshift(d, pos_indices[j])
                redshift_mat[i, j] = rs

                # Causal mask
                cm = self.causal_mask(
                    token_positions[i],
                    token_positions[j],
                    pos_indices[j]
                )
                causal_mat[i, j] = cm

        # Apply: score × redshift × causal_mask
        constrained = base_scores * redshift_mat * causal_mat

        return constrained

    def forward(self, token_vecs: torch.Tensor,
                token_positions: list,
                pos_indices: list,
                base_scores: torch.Tensor = None
                ) -> tuple:
        """
        Full light-speed constrained attention.

        Returns:
            attended: (L, d)
            constrained_scores: (L, L)
            photon_mask: (L,) — which tokens are photons
        """
        L = token_vecs.shape[0]
        device = token_vecs.device

        if base_scores is None:
            base_scores = torch.ones(L, L, device=device)

        # Identify photon tokens (massless)
        photon_mask = torch.tensor(
            [self.get_mass(p) < 0.01 for p in pos_indices],
            dtype=torch.float, device=device
        )

        # Compute constrained scores
        scores = self.compute_light_speed_scores(
            token_vecs, token_positions,
            pos_indices, base_scores
        )

        # Photons bypass causal mask — they travel at c
        # and can see full past
        for i in range(L):
            if photon_mask[i] > 0.5:
                for j in range(i):
                    d = abs(token_positions[i] - token_positions[j])
                    rs = self.redshift(d, pos_indices[j])
                    scores[i, j] = base_scores[i, j] * rs

        # Softmax attention
        scores_safe = scores.clone()
        scores_safe[scores_safe == 0] = -1e9
        A = torch.softmax(scores_safe, dim=-1)
        A[scores == 0] = 0  # zero out masked positions

        # Apply redshift to value vectors
        V_redshifted = self.W_redshift(token_vecs)
        attended = A @ V_redshifted

        return attended, scores, photon_mask


def compute_orbital_redshift(ring_depth_i: int,
                              ring_depth_j: int,
                              c_orbital: float = 3.0) -> float:
    """
    Orbital redshift between rings at different depths.
    Information from deeper rings (moons) arrives
    faded at the sun level.

    λ_orbital = e^(-depth_diff / c_orbital)
    """
    depth_diff = abs(ring_depth_i - ring_depth_j)
    return math.exp(-depth_diff / c_orbital)
