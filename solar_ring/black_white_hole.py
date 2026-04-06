import torch
import torch.nn as nn
from solar_ring.sun_state import SunState

# Event horizon threshold
EVENT_HORIZON = 0.1

# White hole spawn threshold
WHITE_HOLE_THRESHOLD = 0.7

# Hawking radiation strength
HAWKING_STRENGTH = 0.3

# Contradiction tokens that trigger black hole
NEGATION_TOKENS = {
    'not','never','no','none','nothing','nobody',
    'neither','nor','false','wrong','incorrect',
    'denied','refused','impossible','contradicted'
}

# Question tokens that trigger white hole
QUESTION_TOKENS = {
    'who','what','which','where','when','why','how',
    'whose','whom'
}


class RingConfidence:
    """
    Tracks confidence level of each ring.
    Confidence decays over time.
    Falls below event horizon → black hole collapse.
    """
    def __init__(self, initial=1.0, decay=0.05):
        self.value = initial
        self.decay_rate = decay
        self.age = 0

    def decay(self):
        """Exponential decay each token step."""
        self.value *= (1.0 - self.decay_rate)
        self.age += 1

    def boost(self, amount=0.2):
        """Boost when ring is accessed/referenced."""
        self.value = min(1.0, self.value + amount)

    def is_collapsed(self):
        return self.value < EVENT_HORIZON

    def __repr__(self):
        return f"conf={self.value:.3f} age={self.age}"


class BlackHole:
    """
    Collapses a ring when confidence falls below
    event horizon. Transfers compressed info to Sun State
    as Hawking radiation before collapse.
    """
    def __init__(self, sun_state: SunState):
        self.sun_state = sun_state
        self.collapsed_rings = []  # history of collapses

    def check_collapse(self, ring, confidence: RingConfidence,
                       ring_idx: int) -> bool:
        """
        Check if ring should collapse.
        If yes: emit Hawking radiation to Sun State.
        Returns True if ring collapsed.
        """
        if not confidence.is_collapsed():
            return False

        # Emit Hawking radiation before collapse
        # Compressed summary of ring → Sun State
        self._emit_hawking(ring)

        # Record collapse
        self.collapsed_rings.append({
            'ring_idx': ring_idx,
            'age': confidence.age,
            'confidence': confidence.value,
        })

        return True

    def _emit_hawking(self, ring):
        """
        Transfer compressed ring info to Sun State.
        Information appears destroyed but echoes persist.
        """
        radiation = []
        if hasattr(ring, 'lock_s') and ring.lock_s:
            radiation.append(ring.s * HAWKING_STRENGTH)
        if hasattr(ring, 'lock_o') and ring.lock_o:
            radiation.append(ring.o * HAWKING_STRENGTH)
        if hasattr(ring, 'v') and ring.v.norm() > 0:
            radiation.append(ring.v * HAWKING_STRENGTH)

        if radiation:
            self.sun_state.fuse(radiation)

    def check_contradiction(self, token_text: str,
                            ring, confidence: RingConfidence,
                            ring_idx: int) -> bool:
        """
        Negation token near active ring → forced collapse.
        """
        if token_text.lower() in NEGATION_TOKENS:
            # Accelerate decay to event horizon
            confidence.value *= 0.3
            return self.check_collapse(ring, confidence, ring_idx)
        return False

    def check_topic_shift(self, ring, confidence: RingConfidence,
                          sun_state: SunState,
                          ring_idx: int) -> bool:
        """
        New paragraph with low resonance → collapse.
        """
        if not hasattr(ring, 's'):
            return False

        res = sun_state.resonance(ring.s)
        if res < 0.1 and confidence.age > 10:
            confidence.value *= 0.5
            return self.check_collapse(ring, confidence, ring_idx)
        return False


class WhiteHole:
    """
    Spawns new rings from nothing when needed.
    Triggered by orphaned pronouns, new entities,
    questions, or resolved contradictions.
    """
    def __init__(self, d_model: int, device,
                 sun_state: SunState):
        self.d = d_model
        self.device = device
        self.sun_state = sun_state
        self.spawned_rings = []

    def check_orphan_pronoun(self, token_text: str,
                              memory) -> bool:
        """
        Pronoun with no matching antecedent found.
        Spawn a placeholder ring seeded by Sun State.
        """
        PRONOUNS = {'it','he','she','they','him',
                    'her','them','who','which','that'}

        if token_text.lower() not in PRONOUNS:
            return False

        # Check if any ring has a subject
        has_antecedent = any(
            ring.lock_s for ring in memory.rings
            if hasattr(ring, 'lock_s')
        )

        if not has_antecedent:
            self._spawn_placeholder(memory, token_text)
            return True

        return False

    def check_question(self, token_text: str,
                       memory) -> bool:
        """
        Question word detected → spawn answer-seeking ring.
        This ring gravitates toward the answer.
        """
        if token_text.lower() not in QUESTION_TOKENS:
            return False

        self._spawn_question_ring(memory, token_text)
        return True

    def _spawn_placeholder(self, memory, pronoun_text: str):
        """
        Spawn placeholder ring seeded by Sun State context.
        The ring starts with Sun State's current knowledge
        as its subject pole — best guess at antecedent.
        """
        new_ring_idx = len(memory.rings)
        if new_ring_idx >= 13:
            return  # max rings reached

        # Seed with Sun State knowledge
        seed_vec = self.sun_state.state.clone()
        if seed_vec.norm() < 1e-6:
            seed_vec = torch.randn(self.d, device=self.device)
            seed_vec = seed_vec / seed_vec.norm()

        from solar_ring.ring_node import RingNode
        new_id = len(memory.rings)
        placeholder = RingNode(
            device=self.device,
            ring_id=new_id,
            parent_id=new_id - 1,
        )
        placeholder.write_subject(seed_vec)
        memory.rings.append(placeholder)

        self.spawned_rings.append({
            'type': 'placeholder',
            'pronoun': pronoun_text,
            'ring_idx': new_ring_idx,
        })

    def _spawn_question_ring(self, memory, question_word: str):
        """
        Spawn answer-seeking ring.
        Seeded with Sun State — gravitates toward answer.
        """
        self.spawned_rings.append({
            'type': 'question',
            'word': question_word,
        })


class BlackWhiteHoleManager:
    """
    Manages all black holes and white holes in a solar system.
    Called each token step to check and trigger events.
    """
    def __init__(self, d_model: int, device,
                 sun_state: SunState):
        self.d = d_model
        self.device = device
        self.black_hole = BlackHole(sun_state)
        self.white_hole = WhiteHole(d_model, device, sun_state)
        self.confidences = {}  # ring_idx → RingConfidence
        self.collapsed_indices = set()

    def get_confidence(self, ring_idx: int) -> RingConfidence:
        if ring_idx not in self.confidences:
            self.confidences[ring_idx] = RingConfidence()
        return self.confidences[ring_idx]

    def step(self, token_text: str, memory,
             sun_state: SunState):
        """
        Called for every token.
        Updates all ring confidences.
        Checks for black/white hole events.
        Returns list of events that occurred.
        """
        events = []

        # Decay all ring confidences
        for i in range(len(memory.rings)):
            if i not in self.collapsed_indices:
                conf = self.get_confidence(i)
                conf.decay()

                # Boost active ring
                if i == memory.alpha:
                    conf.boost(0.3)

        # Check for black hole collapses
        for i, ring in enumerate(memory.rings):
            if i in self.collapsed_indices:
                continue
            conf = self.get_confidence(i)

            # Contradiction check
            if self.black_hole.check_contradiction(
                token_text, ring, conf, i
            ):
                self.collapsed_indices.add(i)
                events.append(f"BLACK_HOLE: ring {i} collapsed (contradiction)")
                continue

            # Normal collapse check
            if self.black_hole.check_collapse(ring, conf, i):
                self.collapsed_indices.add(i)
                events.append(f"BLACK_HOLE: ring {i} collapsed (confidence)")

        # Check for white hole spawns
        if self.white_hole.check_question(token_text, memory):
            events.append(f"WHITE_HOLE: question ring spawned for '{token_text}'")

        if self.white_hole.check_orphan_pronoun(token_text, memory):
            events.append(f"WHITE_HOLE: placeholder spawned for '{token_text}'")

        return events

    def summary(self) -> str:
        total = len(self.confidences)
        collapsed = len(self.collapsed_indices)
        spawned = len(self.white_hole.spawned_rings)
        return (f"Rings: {total} total, "
                f"{collapsed} collapsed (black holes), "
                f"{spawned} spawned (white holes)")
