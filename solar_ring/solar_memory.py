"""SolarMemory: manages the full ring hierarchy for one sequence."""

import torch
import torch.nn.functional as F
from .config import (
    D_MODEL, MAX_RINGS, SLOTS_PER_RING, FLAT_SIZE,
    ROLE_SUBJ, ROLE_OBJ, ROLE_VERB, ROLE_CONJ,
    PRONOUN_RECENCY_DECAY, PRONOUN_CONF_SUBJ, PRONOUN_CONF_OBJ,
    PRONOUN_OBJ_PRIOR, PRONOUN_SUN_PRIOR,
)
from .ring_node import RingNode
from .sun_state import SunState

# ── Gender / animacy knowledge for pronoun resolution ───────────────────────
_MALE_WORDS = {
    'man','men','boy','boys','father','son','brother','uncle','grandfather',
    'husband','king','prince','actor','john','paul','tom','mike','sam','bob',
    'steve','jake','chris','alex','george','mark','david','nick','tim','rob',
    'dave','james','peter','henry','edward','william','andrew',
}
_FEMALE_WORDS = {
    'woman','women','girl','girls','mother','daughter','sister','aunt',
    'grandmother','wife','queen','princess','actress','nurse','mary','joan',
    'susan','anna','beth','sara','carol','emma','lisa','amy','rachel','diana',
    'alice','linda','sarah','nina','helen','kate','jane','emily','sophie',
}
_PLURAL_NOUNS = {
    'people','students','workers','children','teachers','protesters','police',
    'managers','doctors','nurses','rebels','scientists','soldiers','army',
    'team','players','animals','creatures','citizens','residents','employees',
}
_INANIMATE_WORDS = {
    'trophy','suitcase','ball','window','box','bottle','book','bag','vase',
    'cup','glass','tile','car','door','table','chair','rock','hammer','pipe',
    'tree','fence','jar','bucket','plate','mat','letter','package','hole',
    'phone','laptop','computer','machine','device','tool','trophy','coin',
    'cat','dog','hawk','rabbit','wolf','deer','bacteria','chicken',
    'sand','water','oil','rain','snow','wind',
}

_PRONOUN_GENDER: dict = {
    'he': 'male',   'him': 'male',   'his': 'male',
    'she': 'female', 'her': 'female', 'hers': 'female',
    'it':  'neutral', 'its': 'neutral',
    'they':'plural', 'them':'plural', 'their':'plural',
}


def _word_gender(word: str) -> str | None:
    """Return gender category for a word, or None if unknown."""
    w = word.lower().strip(".,!?;:'\"()-")
    if w in _MALE_WORDS:    return 'male'
    if w in _FEMALE_WORDS:  return 'female'
    if w in _PLURAL_NOUNS:  return 'plural'
    if w in _INANIMATE_WORDS: return 'neutral'
    return None


def _gender_score(candidate_word: str, pronoun_word: str) -> float:
    """Multiplicative agreement score between candidate entity and pronoun."""
    cg = _word_gender(candidate_word)
    pg = _PRONOUN_GENDER.get(pronoun_word.lower().strip(".,!?;:'\"-"), None)
    if cg is None or pg is None:
        return 1.0   # no info → neutral
    if cg == pg:
        return 2.5   # strong agreement
    if pg == 'plural':
        return 1.3   # 'they' can refer to groups of any gender
    if pg == 'neutral' and cg in ('male', 'female'):
        return 0.05  # 'it' very rarely refers to a person
    if pg in ('male', 'female') and cg == 'neutral':
        return 0.3   # he/she rarely refers to an inanimate thing
    if pg in ('male', 'female') and cg not in (pg, 'neutral', 'plural', None):
        return 0.05  # he ↔ female name: very unlikely
    return 1.0


class SolarMemory:
    """
    Manages a list of up to MAX_RINGS RingNodes.

    Hierarchy:
        rings[0] = sun (main clause)
        rings[1..4] = planets (depth=1)
        rings[5..12] = moons (depth=2)

    alpha: index of the currently active ring.
    """

    def __init__(self, device="cpu", dtype=torch.bfloat16, hard_lock: bool = False):
        # Normalise device to torch.device so str "cuda"/"cpu" both work
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.hard_lock = hard_lock  # False=soft (training), True=hard (inference)

        # Initialise with the sun ring
        sun = RingNode(device=device, dtype=dtype, ring_id=0, parent_id=None)
        sun.depth = 0
        self.rings = [sun]
        self.alpha = 0  # active ring pointer

        # Global document-level memory — persists across clauses/sentences
        self.sun_state = SunState(D_MODEL, alpha=0.3, device=device)

    # ------------------------------------------------------------------
    # Ring management
    # ------------------------------------------------------------------

    def _spawn(self, parent_id: int) -> int:
        """
        Spawn a new child ring. Returns new ring id, or -1 if at capacity.
        """
        if len(self.rings) >= MAX_RINGS:
            return -1  # capacity reached, stay in current ring

        new_id = len(self.rings)
        parent = self.rings[parent_id]
        child = RingNode(
            device=self.device, dtype=self.dtype,
            ring_id=new_id, parent_id=parent_id
        )
        child.depth = parent.depth + 1
        self.rings.append(child)
        return new_id

    def activate(self, ring_id: int):
        self.alpha = ring_id

    @property
    def active_ring(self) -> RingNode:
        return self.rings[self.alpha]

    # ------------------------------------------------------------------
    # Token processing entry point
    # ------------------------------------------------------------------

    def process_token(self, vec: torch.Tensor, role_id: int,
                      verb_gate: torch.Tensor = None,
                      spawn: bool = False,
                      token_text: str = "") -> int:
        """
        Route vec into the active ring based on role_id.
        If spawn=True, create a child ring and switch to it first.
        Returns the active ring index after processing.
        """
        if spawn and role_id == ROLE_CONJ:
            new_id = self._spawn(self.alpha)
            if new_id != -1:
                self.alpha = new_id

        ring = self.active_ring

        if role_id == ROLE_SUBJ:
            ring.write_subject(vec, hard_lock=self.hard_lock)
            if token_text:
                ring.subj_word = token_text
        elif role_id == ROLE_OBJ:
            ring.write_object(vec, hard_lock=self.hard_lock)
            if token_text:
                ring.obj_word = token_text
        elif role_id == ROLE_VERB:
            gate = verb_gate if verb_gate is not None else torch.tensor(0.5, dtype=self.dtype, device=self.device)
            ring.write_verb(vec, gate)
        else:
            ring.write_rotating(vec)

        return self.alpha

    # ------------------------------------------------------------------
    # Pronoun resolution
    # ------------------------------------------------------------------

    def resolve_pronoun(self, x: torch.Tensor, pronoun_word: str = None) -> torch.Tensor:
        """
        Enhanced pronoun resolution with:
        - Gender/animacy agreement (he→male, she→female, it→inanimate, they→plural)
        - Recency decay (closer ancestors score higher)
        - Confidence boost for write-once locked slots
        - OBJ slot candidates in addition to SUBJ
        - Sun State as global context fallback

        x: (D_MODEL,) pronoun query vector
        pronoun_word: actual pronoun text ('he','she','it','they', etc.)
        """
        # Collect ancestors including current
        path = []
        cur = self.alpha
        while cur is not None:
            path.append(cur)
            cur = self.rings[cur].parent_id

        if not path:
            return x

        candidate_vecs:   list[torch.Tensor] = []
        candidate_priors: list[float]        = []

        for dist, rid in enumerate(path):
            ring    = self.rings[rid]
            recency = PRONOUN_RECENCY_DECAY ** dist  # exponential recency decay

            # ── SUBJ candidate ──────────────────────────────────────────
            subj = ring.subject_vector()
            if subj.norm().item() > 0.05:
                conf = PRONOUN_CONF_SUBJ if ring.subj_locked else 1.0
                ga   = _gender_score(ring.subj_word, pronoun_word) if pronoun_word else 1.0
                candidate_vecs.append(subj)
                candidate_priors.append(recency * conf * ga)

            # ── OBJ candidate ───────────────────────────────────────────
            obj = ring.object_vector()
            if obj.norm().item() > 0.05:
                conf = PRONOUN_CONF_OBJ if ring.obj_locked else 1.0
                ga   = _gender_score(ring.obj_word, pronoun_word) if pronoun_word else 1.0
                candidate_vecs.append(obj)
                candidate_priors.append(recency * conf * ga * PRONOUN_OBJ_PRIOR)

        # ── Sun State as global context fallback ─────────────────────────
        if self.sun_state.state.norm().item() > 0.05:
            candidate_vecs.append(self.sun_state.state.to(x.dtype))
            candidate_priors.append(PRONOUN_SUN_PRIOR)

        if not candidate_vecs:
            return x

        C      = torch.stack(candidate_vecs, dim=0)                      # (k, D)
        priors = torch.tensor(candidate_priors, device=x.device, dtype=x.dtype)  # (k,)

        # Attention score + log-prior combination
        scale      = D_MODEL ** 0.5
        dot_scores = (x @ C.T) / scale                                   # (k,)
        log_priors = torch.log(priors.clamp(min=1e-6))
        combined   = dot_scores + log_priors                             # (k,)
        weights    = F.softmax(combined, dim=0)                          # (k,)

        return (weights.unsqueeze(-1) * C).sum(dim=0)                   # (D,)

    # ------------------------------------------------------------------
    # Flatten to fixed size
    # ------------------------------------------------------------------

    def flatten(self) -> torch.Tensor:
        """
        Return always-fixed (MAX_RINGS * SLOTS_PER_RING * D_MODEL,) tensor.
        Pads with zeros for unused rings.
        """
        parts = []
        for i in range(MAX_RINGS):
            if i < len(self.rings):
                parts.append(self.rings[i].to_vector())  # (8, D_MODEL)
            else:
                parts.append(torch.zeros(SLOTS_PER_RING, D_MODEL,
                                         device=self.device, dtype=self.dtype))  # padded on device
        stacked = torch.stack(parts, dim=0)   # (13, 8, D_MODEL)
        return stacked.reshape(-1)            # (53248,)

    def get_summary_vectors(self) -> torch.Tensor:
        """
        Return (MAX_RINGS, D_MODEL) summary vectors (one per ring).
        Used for cross-ring attention.
        """
        parts = []
        for i in range(MAX_RINGS):
            if i < len(self.rings):
                parts.append(self.rings[i].summary_vector())
            else:
                parts.append(torch.zeros(D_MODEL, device=self.device, dtype=self.dtype))  # on device
        return torch.stack(parts, dim=0)  # (13, D_MODEL)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sun State integration
    # ------------------------------------------------------------------

    def end_clause(self):
        """
        Call at end of each sentence/clause.
        Fuses active planet heads into the global Sun State.
        """
        planet_heads = []
        for ring in self.rings:
            sv = ring.summary_vector()
            if sv.norm() > 0:
                planet_heads.append(sv)
        self.sun_state.fuse(planet_heads)

    def get_sun_resonance(self, token_vec: torch.Tensor) -> float:
        """How strongly does token_vec resonate with accumulated Sun memory?"""
        return self.sun_state.resonance(token_vec)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        sun = RingNode(device=self.device, dtype=self.dtype, ring_id=0, parent_id=None)
        sun.depth = 0
        self.rings = [sun]
        self.alpha = 0
        self.sun_state = SunState(D_MODEL, alpha=0.3, device=self.device)

    def __len__(self):
        return len(self.rings)

    def __repr__(self):
        return (f"SolarMemory(rings={len(self.rings)}/{MAX_RINGS}, "
                f"alpha={self.alpha})")

    def print_rings(self):
        """Pretty-print ring contents for debugging."""
        labels = ["SUBJ", "OBJ ", "VERB", "ROT0", "ROT1", "ROT2", "ROT3", "ROT4"]
        depth_names = {0: "SUN", 1: "PLANET", 2: "MOON"}
        for ring in self.rings:
            dn = depth_names.get(ring.depth, f"DEPTH{ring.depth}")
            print(f"\n--- Ring {ring.ring_id} [{dn}] parent={ring.parent_id} ---")
            for i in range(8):
                norm = ring.slot_norm(i)
                locked = ""
                if i == 0:
                    locked = " [LOCKED]" if ring.subj_locked else " [free]"
                elif i == 1:
                    locked = " [LOCKED]" if ring.obj_locked else " [free]"
                print(f"  slot[{i}] {labels[i]}: norm={norm:.4f}{locked}")
