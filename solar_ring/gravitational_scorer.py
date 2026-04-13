"""
Gravitational Scorer — pure physics math.

Every entity in a sentence emits gravitational force.
Attraction: cosine similarity > 0 → entities pull together
Repulsion:  cosine similarity < 0 → entities push apart
Neutral:    cosine similarity = 0 → no interaction

Score formula:
  F(i,j) = G × m_i × m_j × cos(v_i, v_j) / r²

Where:
  G   = gravitational constant (1.0)
  m_i = semantic mass of entity i (by POS type)
  r   = positional distance between tokens
  cos = cosine similarity of embeddings

Total gravitational score for candidate C in context S:
  Φ(C|S) = Σ F(C, token_k) for all k in S

Higher Φ = stronger gravitational pull toward context
→ C is more likely the correct referent
"""

import torch
import torch.nn as nn
import math


class GravitationalScorer(nn.Module):
    """
    Scores entities using gravitational physics.
    Attraction and repulsion based on semantic similarity.
    """

    # Semantic mass by word type
    MASS = {
        'NOUN':  1.0,
        'VERB':  0.8,
        'ADJ':   0.6,
        'ADV':   0.4,
        'PRON':  0.0,  # massless — photon
        'DET':   0.05,
        'PREP':  0.1,
        'CONJ':  0.1,
        'OTHER': 0.3,
    }

    # POS detection by word
    VERBS = {'is','was','were','are','be','been','being',
             'has','have','had','do','does','did',
             'chased','broke','fell','burst','filled',
             'crushed','overflowed','gave','told','helped',
             'obeyed','struck','fit','flooded','caused'}
    ADJS  = {'big','small','large','tiny','heavy','light',
             'strong','weak','fast','slow','hungry','tired',
             'angry','happy','sad','aggressive','sharp',
             'narrow','wide','tall','short','hot','cold',
             'warm','hard','soft','loud','quiet','scared'}
    PRONOUNS = {'it','its','he','him','his','she','her',
                'they','them','their','who','which','that'}
    DETS  = {'the','a','an','this','that','these','those',
             'my','your','his','her','its','our','their'}
    PREPS = {'in','on','at','by','for','with','from','to',
             'of','about','through','until','because','since'}
    CONJS = {'and','but','or','so','yet','nor','although',
             'because','since','while','when','if','unless'}

    # Semantic roles that REPEL each other
    AGENT_WORDS = {
        # Predators / active animals
        'wolf','hawk','cat','lion','fox','eagle','shark',
        'bear','cheetah','dog','tiger','leopard','crocodile',
        # Authority figures
        'manager','managers','teacher','teachers','parent',
        'parents','director','directors','doctor','doctors',
        'boss','police','officer','captain','leader','chief',
        'employer','employers','supervisor','supervisors',
        'principal','principals','commander','commanders',
        'instructor','instructors','coach','coaches',
        'judge','judges','administrator',
        # Physical agents (cause damage)
        'hammer','ball','rock','stone','truck','axe','knife',
        'boot','beam','boulder','tree','pipe','dam','tank',
        'hose','boiler',
    }

    PATIENT_WORDS = {
        # Prey animals
        'deer','rabbit','mouse','zebra','chicken','fish',
        'seal','salmon','gazelle','squirrel','bird','lamb',
        # Subordinates
        'worker','workers','student','students','employee',
        'employees','child','children','patient','patients',
        'subordinate','subordinates','trainee','trainees',
        'recruit','recruits','intern','interns',
        # Physical patients (receive damage)
        'window','tile','hole','vase','box','glass','bread',
        'car','bicycle','tent','basement','valley','floor',
        'garden','room','wall',
        # Containers that overflow
        'bucket','cup','mug','pond',
    }

    # Causal keywords and what they imply
    CAUSE_IMPLIES_AGENT = {
        # Physical/predator properties
        'hungry','aggressive','strong','heavy','sharp',
        'fast','powerful','angry','dominant','active',
        # Authority/command signals
        'obeyed','orders','commanded','instructed','directed',
        'rules','informative','strict','experienced','clear',
        'gave','told','set',
    }
    CAUSE_IMPLIES_PATIENT = {
        'scared','afraid','weak','fragile','narrow',
        'small','crushed','damaged','broken','overflowed',
        'flooded','injured','hurt','destroyed',
    }

    def __init__(self, d: int = 384, G: float = 1.0):
        super().__init__()
        self.d = d

        # Learned gravitational constant
        self.log_G = nn.Parameter(torch.tensor(math.log(G)))

        # Learned mass scaling per type
        self.mass_scale = nn.Parameter(torch.ones(8))

        # Attraction/repulsion threshold
        self.threshold = nn.Parameter(torch.tensor(0.0))

        # Projection for better similarity
        self.W = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.W.weight)

    def G_const(self) -> float:
        return torch.exp(self.log_G).item()

    def get_pos(self, word: str) -> str:
        w = word.lower().rstrip('.,;:!?')
        if w in self.PRONOUNS: return 'PRON'
        if w in self.DETS:     return 'DET'
        if w in self.VERBS:    return 'VERB'
        if w in self.ADJS:     return 'ADJ'
        if w in self.PREPS:    return 'PREP'
        if w in self.CONJS:    return 'CONJ'
        return 'NOUN'

    def get_mass(self, word: str) -> float:
        pos = self.get_pos(word)
        base = self.MASS.get(pos, 0.3)
        # Scale by word length (longer = more specific = heavier)
        length_factor = min(len(word) / 10.0, 1.0)
        return base * (0.7 + 0.3 * length_factor)

    def cosine_sim(self,
                   v1: torch.Tensor,
                   v2: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity between two vectors.
        Positive → attraction
        Negative → repulsion
        """
        v1p = self.W(v1)
        v2p = self.W(v2)
        n1 = v1p.norm()
        n2 = v2p.norm()
        if n1 < 1e-8 or n2 < 1e-8:
            return torch.tensor(0.0, device=v1.device)
        return torch.dot(v1p, v2p) / (n1 * n2)

    def gravitational_force(self,
                             v_i: torch.Tensor,
                             v_j: torch.Tensor,
                             mass_i: float,
                             mass_j: float,
                             r: float) -> float:
        """
        F(i,j) = G × m_i × m_j × cos(v_i,v_j) / r²

        Positive F = attraction (similar entities)
        Negative F = repulsion (dissimilar entities)
        """
        G = self.G_const()
        r_safe = max(r, 1.0)
        cos = self.cosine_sim(v_i, v_j)

        F = G * mass_i * mass_j * cos / (r_safe ** 2)
        return F.item() if isinstance(F, torch.Tensor) else F

    def semantic_role_force(self,
                             cand_word: str,
                             context_words: list) -> float:
        """
        Compute role-based force from semantic context.
        Agent words in agent context → attraction.
        Patient words in agent context → repulsion.
        """
        cw = cand_word.lower().rstrip('.,;')
        is_agent   = cw in self.AGENT_WORDS
        is_patient = cw in self.PATIENT_WORDS

        if not is_agent and not is_patient:
            return 0.0

        ctx_lower = ' '.join(context_words).lower()

        agent_signals = sum(
            1 for w in self.CAUSE_IMPLIES_AGENT
            if w in ctx_lower
        )
        patient_signals = sum(
            1 for w in self.CAUSE_IMPLIES_PATIENT
            if w in ctx_lower
        )

        role_score = agent_signals - patient_signals

        if is_agent   and role_score > 0: return +2.0
        if is_patient and role_score > 0: return -2.0
        if is_agent   and role_score < 0: return -1.0
        if is_patient and role_score < 0: return +1.0

        return 0.0

    def container_force(self,
                         cand_word: str,
                         context_words: list) -> float:
        """
        Containers overflow; liquids cause overflow.
        Detects overflow/burst context and adjusts sign.
        """
        cw  = cand_word.lower().rstrip('.,;')
        ctx = ' '.join(context_words).lower()

        OVERFLOW_VERBS = {'overflowed','overflow','filled',
                          'flooded','burst','leaked','sprayed'}
        CONTAINERS     = {'bucket','glass','cup','mug','pond',
                          'tank','pipe','dam','hose','barrel'}
        LIQUIDS        = {'water','juice','rain','milk','beer',
                          'oil','blood','lava','flood'}

        has_overflow = any(v in ctx for v in OVERFLOW_VERBS)
        if not has_overflow:
            return 0.0

        if cw in CONTAINERS: return +3.0
        if cw in LIQUIDS:    return -2.0

        return 0.0

    def score_candidate(self,
                         candidate_vec: torch.Tensor,
                         candidate_word: str,
                         context_vecs: list,
                         context_words: list,
                         candidate_pos: int) -> dict:
        """
        Compute total gravitational score for a candidate
        entity in context.

        Returns dict with:
          total_phi: total gravitational potential
          attraction: sum of attractive forces
          repulsion:  sum of repulsive forces
          net_force:  attraction - repulsion
          confidence: how certain the score is
        """
        c_mass = self.get_mass(candidate_word)

        total_phi      = 0.0
        attraction     = 0.0
        repulsion      = 0.0
        n_interactions = 0

        for j, (ctx_vec, ctx_word) in enumerate(
            zip(context_vecs, context_words)
        ):
            ctx_mass = self.get_mass(ctx_word)

            # Skip massless entities (pronouns)
            if ctx_mass < 0.01:
                continue

            # Positional distance
            r = abs(candidate_pos - j) + 1

            # Gravitational force (cosine-based)
            F = self.gravitational_force(
                candidate_vec, ctx_vec,
                c_mass, ctx_mass, r
            )

            # Semantic role force (agent/patient context)
            role_F = self.semantic_role_force(
                candidate_word, context_words
            )
            F += role_F

            total_phi += F
            if F > 0:
                attraction += F
            else:
                repulsion += abs(F)

            n_interactions += 1

        # Container/overflow force (applied once, not per token)
        total_phi += self.container_force(
            candidate_word, context_words
        )

        net        = attraction - repulsion
        confidence = abs(net) / max(n_interactions, 1)

        return {
            'total_phi':      total_phi,
            'attraction':     attraction,
            'repulsion':      repulsion,
            'net_force':      net,
            'confidence':     confidence,
            'mass':           c_mass,
            'n_interactions': n_interactions,
        }

    def resolve_pronoun(self,
                         sentence: str,
                         candidates: list,
                         embedder,
                         device) -> dict:
        """
        Resolve pronoun to best candidate using gravity.

        candidates: list of (word, position) tuples
        Returns best candidate with full physics breakdown.
        """
        words = sentence.lower().split()

        # Get embeddings for all words
        with torch.no_grad():
            vecs = embedder.embed_words(sentence).to(device)

        if vecs.shape[0] != len(words):
            if vecs.shape[0] < len(words):
                pad = torch.zeros(
                    len(words) - vecs.shape[0], self.d
                ).to(device)
                vecs = torch.cat([vecs, pad])
            else:
                vecs = vecs[:len(words)]

        results = []
        for cand_word, cand_pos in candidates:
            cand_pos = min(cand_pos, len(words) - 1)
            cand_vec = vecs[cand_pos]

            score = self.score_candidate(
                cand_vec, cand_word,
                [vecs[j] for j in range(len(words))],
                words,
                cand_pos
            )
            score['word'] = cand_word
            score['pos']  = cand_pos
            results.append(score)

        # Best = highest total gravitational potential
        best = max(results, key=lambda x: x['total_phi'])

        return {
            'best':   best,
            'all':    results,
            'winner': best['word'],
            'margin': (best['total_phi'] -
                       min(r['total_phi'] for r in results)
                       if len(results) > 1 else 0),
        }

    def forward(self,
                 sent_c: str,
                 sent_w: str,
                 embedder,
                 device) -> tuple:
        """
        Compare two sentences using gravitational scoring.
        Returns (score_correct, score_wrong).
        Higher score = stronger gravitational alignment.
        """
        def sentence_phi(sentence):
            words = sentence.lower().split()
            with torch.no_grad():
                vecs = embedder.embed_words(sentence).to(device)

            if vecs.shape[0] < len(words):
                pad = torch.zeros(
                    len(words) - vecs.shape[0], self.d
                ).to(device)
                vecs = torch.cat([vecs, pad])
            else:
                vecs = vecs[:len(words)]

            total = 0.0
            n = len(words)
            for i in range(n):
                for j in range(i + 1, n):
                    mi = self.get_mass(words[i])
                    mj = self.get_mass(words[j])
                    if mi < 0.01 or mj < 0.01:
                        continue
                    r = j - i
                    F = self.gravitational_force(
                        vecs[i], vecs[j],
                        mi, mj, r
                    )
                    total += F
            return total

        phi_c = sentence_phi(sent_c)
        phi_w = sentence_phi(sent_w)
        return phi_c, phi_w
