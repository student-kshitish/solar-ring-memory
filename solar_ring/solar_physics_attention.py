"""Solar Physics Attention — gravitational scoring between orbital concepts.

Each token / ring is modelled as a celestial body:
  mass         = embedding L2 norm          (content richness)
  radius       = 3.0 ** dep_depth           (orbital distance from sun)
  eccentricity = 1.0 − pos_confidence       (orbit irregularity / ambiguity)
  angle_vec    = 8-dim one-hot POS indicator (orbital phase)

Gravitational score between concepts i and j:
  F = G(pi,pj) * m_i*m_j / r_ij^2  ×  cos_angle(vec_i, vec_j)
                                    ×  resonance(pi,pj)
                                    ×  (1 − e_i*e_j)

  • G_matrix   : learnable (8,8) role-pair gravity constants
  • resonance  : learnable (8,8) orbital resonance modifiers
  • cos_angle  : cosine similarity of content vectors (semantic alignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from solar_ring.config import D_MODEL

# Pronoun set — any token whose text appears here gets eccentricity=0.85 (Pluto-class)
PRONOUNS = {
    'it', 'he', 'she', 'they', 'him', 'her', 'them',
    'his', 'hers', 'its', 'their', 'who', 'which', 'that',
}


# ── Orbit class classifier (for display) ─────────────────────────────────────

def orbit_class(pos_type: str, eccentricity: float) -> str:
    """Map (POS role, eccentricity) to a planet name."""
    if eccentricity > 0.70:
        return "Pluto"
    if eccentricity > 0.45:
        return "Neptune"
    if pos_type in ("SUBJ", "VERB") and eccentricity <= 0.12:
        return "Mercury"
    if pos_type == "OBJ" and eccentricity <= 0.12:
        return "Venus"
    if pos_type in ("SUBJ", "VERB") and eccentricity <= 0.35:
        return "Earth"
    if pos_type == "OBJ" and eccentricity <= 0.35:
        return "Mars"
    if pos_type in ("ADJ", "ADV", "CONJ"):
        return "Jupiter"
    return "Saturn"


# ── OrbitalConcept ─────────────────────────────────────────────────────────────

class OrbitalConcept:
    """
    Represents one token (or ring summary) as a celestial body.

    Parameters
    ----------
    token_vec      : (D_MODEL,) float32 embedding on device
    pos_type       : POS role name ('SUBJ','VERB','OBJ','PREP','CONJ','ADJ','ADV','DET')
    dep_depth      : syntactic depth (0=main clause, 1=embedded, 2=doubly-embedded)
    pos_confidence : float in [0,1] — confidence in the POS assignment
    device         : torch device
    """

    _POS_TO_IDX: dict = {
        "SUBJ": 0, "VERB": 1, "OBJ": 2, "PREP": 3,
        "CONJ": 4, "ADJ":  5, "ADV": 6, "DET":  7,
    }

    def __init__(
        self,
        token_vec:      torch.Tensor,
        pos_type:       str,
        dep_depth:      int,
        pos_confidence: float,
        device,
        token_text:     str = "",
    ):
        self.vec      = token_vec.float().to(device)
        self.pos_type = pos_type
        self.mass     = float(token_vec.float().norm().item())
        self.radius   = 3.0 ** dep_depth

        # Pronouns are always Pluto-class (high eccentricity → unstable orbit)
        if token_text.lower() in PRONOUNS:
            self.eccentricity = 0.85
            self.orbit_class  = "Pluto"
        else:
            self.eccentricity = 1.0 - float(pos_confidence)
            self.orbit_class  = None  # determined by orbit_class() fn

        self.angle_vec = torch.zeros(8, device=device, dtype=torch.float32)
        idx = self._POS_TO_IDX.get(pos_type)
        if idx is not None:
            self.angle_vec[idx] = 1.0


# ── SolarPhysicsAttention ──────────────────────────────────────────────────────

class SolarPhysicsAttention(nn.Module):
    """
    Gravitational attention over a sequence of OrbitalConcepts.

    Learnable parameters
    --------------------
    G_matrix  : (8, 8) — role-pair gravity constants, init 0.1
    resonance : (8, 8) — orbital resonance, init eye * 0.5
    W_v       : Linear(d, d) — value projection
    W_out     : Linear(d, d) — output projection

    forward(concepts, token_vecs) → (out, A, scores)
    """

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.d = d_model
        self.G_matrix = nn.Parameter(torch.ones(8, 8) * 0.1)
        self.resonance = nn.Parameter(torch.eye(8) * 0.5)
        self.W_v   = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

    # ------------------------------------------------------------------

    def gravitational_score(
        self,
        ci: OrbitalConcept,
        cj: OrbitalConcept,
    ) -> torch.Tensor:
        """
        Scalar gravitational attraction from concept i toward concept j.

        F = G * m_i * m_j / r_ij^2  ×  cos_angle(vec_i, vec_j)
                                     ×  resonance(pi, pj)
                                     ×  (1 − e_i * e_j)
        """
        r_ij = abs(ci.radius - cj.radius) + 1.0

        pi = int(ci.angle_vec.argmax().item())
        pj = int(cj.angle_vec.argmax().item())

        G    = torch.sigmoid(self.G_matrix[pi, pj].float())
        base = G * ci.mass * cj.mass / (r_ij ** 2)

        # Cosine similarity of content vectors (semantic orbital alignment)
        norm_i    = ci.vec.norm() + 1e-8
        norm_j    = cj.vec.norm() + 1e-8
        cos_angle = torch.dot(ci.vec.to(cj.vec.device), cj.vec) / (norm_i * norm_j)

        res        = torch.sigmoid(self.resonance[pi, pj].float())
        ecc_factor = 1.0 - (ci.eccentricity * cj.eccentricity)

        return base * cos_angle * res * ecc_factor

    # ------------------------------------------------------------------

    def forward(
        self,
        concepts:   list,           # List[OrbitalConcept], length L
        token_vecs: torch.Tensor,   # (L, D_MODEL) float32
    ):
        """
        Compute gravitational attention and return attended representations.

        Returns
        -------
        out    : (L, D_MODEL) — W_out applied to weighted value vectors
        A      : (L, L)       — softmax attention weights
        scores : (L, L)       — raw gravitational scores
        """
        L   = len(concepts)
        dev = token_vecs.device

        # Build (L, L) score matrix — preserve grad through G_matrix & resonance
        score_rows = []
        for i in range(L):
            row = []
            for j in range(L):
                if i != j:
                    s = self.gravitational_score(concepts[i], concepts[j])
                    s = s.to(dev) if isinstance(s, torch.Tensor) \
                        else torch.tensor(float(s), device=dev)
                else:
                    s = torch.zeros((), device=dev)
                row.append(s)
            score_rows.append(torch.stack(row))     # (L,)
        scores = torch.stack(score_rows)            # (L, L)

        A        = torch.softmax(scores.float(), dim=-1)   # (L, L)
        V        = self.W_v(token_vecs.float())            # (L, d)
        attended = A @ V                                    # (L, d)
        out      = self.W_out(attended)                    # (L, d)

        return out, A, scores
