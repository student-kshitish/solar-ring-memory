import torch
import torch.nn as nn
from solar_ring.solar_memory import SolarMemory
from solar_ring.sun_state import SunState
from solar_ring.multi_solar_system import MultiSolarSystem

class GalacticCore:
    """
    Topic-level memory shared across all documents.
    All solar systems in a galaxy orbit this core.
    Very slow fusion rate β=0.1 — topic changes slowly.
    """
    def __init__(self, d_model: int, device,
                 beta: float = 0.1):
        self.d = d_model
        self.device = device
        self.beta = beta
        self.state = torch.zeros(d_model, device=device)
        self.cluster_count = 0

    def fuse_document(self, sun_state_vec: torch.Tensor):
        """
        Absorb a document's Sun State into galactic core.
        Very slow fusion — topic is stable across documents.
        """
        self.state = ((1 - self.beta) * self.state +
                      self.beta * sun_state_vec)
        self.cluster_count += 1

    def topic_similarity(self,
                         sun_vec: torch.Tensor) -> float:
        """
        How similar is this document to the galactic topic?
        High = belongs to this galaxy.
        Low = belongs to different galaxy.
        """
        if self.state.norm() < 1e-6:
            return 0.0
        return torch.nn.functional.cosine_similarity(
            sun_vec.unsqueeze(0),
            self.state.unsqueeze(0)
        ).item()

    def sub_galaxy_gravity(self,
                           sun_i: torch.Tensor,
                           sun_j: torch.Tensor,
                           G_galaxy: float = 0.5) -> float:
        """
        Gravitational attraction between two documents
        within the same galaxy.
        G_sg = G_galaxy × sim(i, core) × sim(j, core)
        """
        sim_i = self.topic_similarity(sun_i)
        sim_j = self.topic_similarity(sun_j)
        return G_galaxy * sim_i * sim_j


class SubGalaxy:
    """
    A cluster of related solar systems (documents)
    within one galaxy. Like a spiral arm.
    """
    def __init__(self, d_model: int, device,
                 galactic_core: GalacticCore):
        self.d = d_model
        self.device = device
        self.core = galactic_core
        self.systems = []       # MultiSolarSystem per doc
        self.sun_vecs = []      # Sun State per document
        self.centroid = torch.zeros(d_model, device=device)

    def add_document(self, mss: MultiSolarSystem):
        """Add a processed document to this sub-galaxy."""
        sun_vec = mss.active_sun.state.clone()
        self.systems.append(mss)
        self.sun_vecs.append(sun_vec)

        # Update centroid
        all_vecs = torch.stack(self.sun_vecs)
        self.centroid = all_vecs.mean(dim=0)

        # Fuse into galactic core
        self.core.fuse_document(sun_vec)

    def inter_cluster_gravity(self,
                              query_vec: torch.Tensor,
                              top_k: int = 3) -> torch.Tensor:
        """
        Find most relevant documents in this cluster
        for a given query vector.
        Returns weighted blend of top-k Sun States.
        """
        if not self.sun_vecs:
            return torch.zeros(self.d, device=self.device)

        scores = []
        for sv in self.sun_vecs:
            g = self.core.sub_galaxy_gravity(query_vec, sv)
            scores.append(g)

        scores_t = torch.tensor(
            scores, device=self.device, dtype=torch.float
        )

        k = min(top_k, len(scores))
        top_vals, top_idx = scores_t.topk(k)
        weights = torch.softmax(top_vals, dim=0)

        result = torch.zeros(self.d, device=self.device)
        for i, idx in enumerate(top_idx):
            result += weights[i] * self.sun_vecs[idx]

        return result


class MultiverseMemory:
    """
    Parallel processing of ambiguous sentences.
    Spawns multiple universes when ambiguity detected.
    Each universe = independent SolarMemory.
    Collapses to highest-probability universe when resolved.

    Like quantum superposition until observation.
    """
    AMBIGUITY_THRESHOLD = 1.5   # entropy threshold
    COLLAPSE_THRESHOLD  = 0.80  # probability to collapse

    AMBIGUOUS_WORDS = {
        'bank': ['river_bank', 'financial_bank'],
        'bat':  ['baseball_bat', 'animal_bat'],
        'bark': ['tree_bark', 'dog_bark'],
        'bear': ['animal_bear', 'bear_verb'],
        'bow':  ['bow_weapon', 'bow_gesture'],
        'can':  ['can_able', 'can_container'],
        'fly':  ['fly_insect', 'fly_verb'],
        'left': ['left_direction', 'left_verb'],
        'light':['light_weight', 'light_illumination'],
        'match':['match_fire', 'match_game'],
        'mean': ['mean_unkind', 'mean_average'],
        'mine': ['mine_possessive', 'mine_excavation'],
        'park': ['park_garden', 'park_verb'],
        'plant':['plant_organism', 'plant_factory'],
        'right':['right_direction', 'right_correct'],
        'rose': ['rose_flower', 'rose_verb'],
        'run':  ['run_verb', 'run_noun'],
        'spring':['spring_season', 'spring_mechanical'],
        'type': ['type_kind', 'type_verb'],
        'wave': ['wave_ocean', 'wave_gesture'],
    }

    def __init__(self, d_model: int, device,
                 galactic_core: GalacticCore = None):
        self.d = d_model
        self.device = device
        self.galactic_core = galactic_core

        # Active universes
        self.universes = []      # list of SolarMemory
        self.probabilities = []  # P per universe
        self.labels = []         # interpretation label
        self.is_collapsed = True # start collapsed
        self.selected = None     # final universe after collapse

    def check_ambiguity(self, token_text: str) -> list:
        """
        Check if token is ambiguous.
        Returns list of interpretations if yes else empty.
        """
        return self.AMBIGUOUS_WORDS.get(
            token_text.lower(), []
        )

    def spawn_universes(self, token_text: str,
                        base_memory: SolarMemory):
        """
        Spawn parallel universes for ambiguous token.
        Each universe = copy of base_memory + different interpretation.
        """
        interpretations = self.check_ambiguity(token_text)
        if not interpretations:
            return False

        self.universes = []
        self.probabilities = []
        self.labels = []

        # Equal probability at spawn
        p = 1.0 / len(interpretations)

        for interp in interpretations:
            # Each universe starts as copy of base memory
            new_mem = SolarMemory(device=self.device)

            # Seed with different interpretation bias
            bias = torch.zeros(self.d, device=self.device)
            wid = hash(interp) % self.d
            bias[wid] = 1.0

            # Write interpretation bias to sun
            new_mem.sun_state = SunState(
                self.d, alpha=0.3, device=self.device
            )
            new_mem.sun_state.state = bias * 0.3

            self.universes.append(new_mem)
            self.probabilities.append(p)
            self.labels.append(interp)

        self.is_collapsed = False
        print(f"  [MULTIVERSE] Spawned {len(interpretations)} "
              f"universes for '{token_text}'")
        return True

    def update_probabilities(self,
                             context_vec: torch.Tensor):
        """
        Update universe probabilities based on new context.
        P(universe_k) = softmax(sun_k · context / √d)
        """
        if self.is_collapsed or not self.universes:
            return

        scores = []
        for mem in self.universes:
            sun = mem.sun_state.state
            score = torch.dot(sun, context_vec)
            score = score / (self.d ** 0.5)
            scores.append(score.item())

        scores_t = torch.tensor(
            scores, device=self.device, dtype=torch.float
        )
        probs = torch.softmax(scores_t, dim=0)
        self.probabilities = probs.tolist()

        # Check if ready to collapse
        if max(self.probabilities) >= self.COLLAPSE_THRESHOLD:
            self.collapse()

    def collapse(self) -> SolarMemory:
        """
        Select highest-probability universe.
        Discard all others.
        Like quantum measurement collapsing wavefunction.
        """
        if self.is_collapsed:
            return self.selected

        best_idx = self.probabilities.index(
            max(self.probabilities)
        )
        self.selected = self.universes[best_idx]
        self.is_collapsed = True

        print(f"  [MULTIVERSE] Collapsed to universe "
              f"'{self.labels[best_idx]}' "
              f"(P={self.probabilities[best_idx]:.3f})")

        return self.selected

    def get_active_memory(self) -> SolarMemory:
        """
        Return active memory.
        If collapsed: return selected universe.
        If not collapsed: return highest-prob universe.
        """
        if self.is_collapsed and self.selected:
            return self.selected
        if self.universes:
            best = self.probabilities.index(
                max(self.probabilities)
            )
            return self.universes[best]
        return None

    def parallel_step_count(self) -> int:
        """How many universes running i."""
        return 0 if self.is_collapsed else len(self.universes)
