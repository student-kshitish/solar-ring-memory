import torch
import torch.nn as nn
import math

class UnifiedLightField(nn.Module):
    """
    One formula for all relationships, reasoning,
    memory, and spatial connections.

    Phi(i,j) = lambda(d) * G(m,r) * C(i,j) * R(i,j)
               * (1-BH_i) * (1-BH_j)

    Positive Phi = attraction
    Negative Phi = repulsion
    Zero Phi = neutral/causal isolation
    """

    # Speed of light per domain (learned)
    C_DOMAINS = {
        'relationship': 50.0,
        'reasoning':    10.0,
        'memory':       50.0,
        'orbital':       3.0,
        'temporal':     20.0,
    }

    # Semantic mass per entity type
    MASS = {
        'self':       1.00,
        'parent':     0.95,
        'child':      0.95,
        'sibling':    0.90,
        'spouse':     0.90,
        'best_friend':0.85,
        'close_friend':0.75,
        'colleague':  0.65,
        'classmate':  0.60,
        'professor':  0.65,
        'acquaintance':0.40,
        'stranger':   0.10,
        'subject':    0.95,
        'verb':       0.85,
        'object':     0.90,
        'adjective':  0.50,
        'article':    0.05,
        'pronoun':    0.00,  # photon — massless
    }

    def __init__(self, d: int = 300):
        super().__init__()
        self.d = d

        # Learned speed of light per domain
        self.log_c = nn.ParameterDict({
            domain: nn.Parameter(
                torch.tensor(math.log(c))
            )
            for domain, c in self.C_DOMAINS.items()
        })

        # Learned gravitational constant
        self.log_G = nn.Parameter(torch.tensor(0.0))

        # Resonance projection
        self.W_res = nn.Linear(d, d)

        # Conflict detector
        self.W_conflict = nn.Linear(d * 2, 1)

    def c(self, domain: str) -> float:
        """Speed of light for domain."""
        if domain in self.log_c:
            return torch.exp(self.log_c[domain]).item()
        return 50.0

    def G_const(self) -> float:
        return torch.exp(self.log_G).item()

    def light_distance(self, d_hops: float,
                       domain: str) -> float:
        """
        Convert hop distance to light-travel time.
        d_light = d_hops / c_domain
        """
        return d_hops / max(self.c(domain), 1e-6)

    def redshift(self, d_light: float,
                 is_photon: bool = False) -> float:
        """
        lambda(d) = e^(-d)
        Photons (pronouns) bypass redshift.
        """
        if is_photon:
            return 1.0
        return math.exp(-d_light)

    def gravity(self, mass_i: float,
                mass_j: float,
                r: float) -> float:
        """
        G * m_i * m_j / r^2
        r = light distance (not hop count)
        """
        G = self.G_const()
        r_safe = max(r, 0.1)
        return G * mass_i * mass_j / (r_safe ** 2)

    def causal_mask(self, pos_i: int,
                    pos_j: int,
                    d_hops: float,
                    domain: str) -> float:
        """
        C(i,j) = 1 if j is in past light cone of i
               = 0 if j is in future or outside cone
        """
        if pos_j >= pos_i:
            return 0.0  # future cannot influence past

        c = self.c(domain)
        travel_time = d_hops / max(c, 1e-6)

        # Within light cone if travel time reasonable
        if travel_time <= 1.0:
            return 1.0

        # Partial inclusion for borderline cases
        return math.exp(-travel_time + 1.0)

    def resonance(self, vec_i: torch.Tensor,
                  vec_j: torch.Tensor) -> float:
        """
        R(i,j) = cos(W_res(i), W_res(j))
        High resonance = semantically aligned
        """
        if vec_i.norm() < 1e-8 or vec_j.norm() < 1e-8:
            return 0.0

        vi = self.W_res(vec_i)
        vj = self.W_res(vec_j)

        return torch.nn.functional.cosine_similarity(
            vi.unsqueeze(0), vj.unsqueeze(0)
        ).item()

    def conflict_score(self, vec_i: torch.Tensor,
                       vec_j: torch.Tensor) -> float:
        """
        Detect contradiction between two vectors.
        High conflict → repulsion (Phi < 0)
        """
        combined = torch.cat([vec_i, vec_j])
        score = torch.sigmoid(
            self.W_conflict(combined.unsqueeze(0))
        ).item()
        return score

    def phi(self,
            entity_i: dict,
            entity_j: dict,
            domain: str = 'relationship') -> float:
        """
        Complete unified field: Phi(i,j)

        entity dict contains:
          'vec':      (d,) embedding vector
          'mass':     float semantic mass
          'pos':      int position/index
          'hops':     float hop distance to j
          'alive':    bool (False = black hole)
          'is_photon': bool

        Returns:
          Phi > 0: attraction
          Phi < 0: repulsion
          Phi = 0: neutral
        """
        # Black hole check — collapsed entities
        BH_i = 0.0 if entity_i.get('alive', True) else 1.0
        BH_j = 0.0 if entity_j.get('alive', True) else 1.0

        if BH_i > 0.5 or BH_j > 0.5:
            return 0.0  # collapsed — no influence

        # Light distance
        d_hops = entity_i.get('hops', 1.0)
        d_light = self.light_distance(d_hops, domain)

        # Photon check
        is_photon = entity_i.get('is_photon', False)

        # Redshift
        lam = self.redshift(d_light, is_photon)

        # Gravity
        mi = entity_i.get('mass', 0.5)
        mj = entity_j.get('mass', 0.5)
        r  = max(d_light, 0.1)
        G_force = self.gravity(mi, mj, r)

        # Causal mask
        pos_i = entity_i.get('pos', 0)
        pos_j = entity_j.get('pos', 0)
        C = self.causal_mask(pos_i, pos_j,
                             d_hops, domain)

        # Resonance
        vec_i = entity_i.get('vec',
                             torch.zeros(self.d))
        vec_j = entity_j.get('vec',
                             torch.zeros(self.d))
        R = self.resonance(vec_i, vec_j)

        # Conflict → repulsion
        conflict = self.conflict_score(vec_i, vec_j)

        if conflict > 0.8:
            # Strong contradiction → repulsion
            return -lam * conflict * mi * mj

        # Unified field
        phi_val = lam * G_force * C * (0.5 + 0.5 * R)

        return phi_val

    def phi_matrix(self, entities: list,
                   domain: str = 'relationship'
                   ) -> torch.Tensor:
        """
        Compute Phi for all entity pairs at once.
        Returns (N, N) matrix.
        """
        N = len(entities)
        matrix = torch.zeros(N, N)

        for i in range(N):
            for j in range(N):
                if i != j:
                    ej = dict(entities[j])
                    ej['hops'] = abs(i - j)
                    ej['pos'] = j
                    ei = dict(entities[i])
                    ei['pos'] = i
                    matrix[i, j] = self.phi(
                        ei, ej, domain
                    )

        return matrix

    def strongest_attraction(self,
                              query: dict,
                              candidates: list,
                              domain: str
                              ) -> tuple:
        """
        Find candidate with strongest Phi to query.
        Used for pronoun resolution, relationship
        finding, and reasoning chain completion.

        Returns (best_candidate, phi_score)
        """
        best_phi = -999
        best_cand = None

        for i, cand in enumerate(candidates):
            c_with_hops = dict(cand)
            c_with_hops['hops'] = abs(
                query.get('pos', 0) - i
            )

            phi_val = self.phi(query, c_with_hops,
                               domain)

            if phi_val > best_phi:
                best_phi = phi_val
                best_cand = cand

        return best_cand, best_phi
