"""SunState: global document-level memory that accumulates across clauses."""

import torch
import torch.nn.functional as F


class SunState:
    """
    Global document-level memory.
    Accumulates knowledge across all clauses and paragraphs.
    Never resets — only fuses new information.

    Formula: Sun(t+1) = (1-α)·Sun(t) + α·mean(Planet slots)
    α = fusion rate (learned or fixed per domain)
    """

    def __init__(self, d_model: int, alpha: float = 0.3, device="cuda"):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.d = d_model
        self.alpha = alpha
        self.device = device
        self.state = torch.zeros(d_model, device=device)
        self.age = 0  # how many fusions have happened

    def fuse(self, planet_slots: list):
        """
        Called at end of each clause/sentence.
        Fuses planet information into sun.
        planet_slots: list of tensors from each planet head
        """
        if not planet_slots:
            return
        planet_mean = torch.stack(planet_slots).float().mean(dim=0).to(self.device)
        self.state = (1 - self.alpha) * self.state + self.alpha * planet_mean
        self.age += 1

    def resonance(self, token_vec: torch.Tensor) -> float:
        """
        Orbital resonance score.
        How strongly does this token resonate with Sun memory?
        High resonance = token is related to established topic.
        Low resonance  = token introduces new information.
        Returns scalar 0.0 to 1.0
        """
        if self.state.norm() < 1e-6:
            return 0.0
        cos_sim = F.cosine_similarity(
            token_vec.float().to(self.device).unsqueeze(0),
            self.state.unsqueeze(0)
        ).item()
        return max(0.0, cos_sim)

    def gravity_pull(self, token_vec: torch.Tensor, pos_mass: float) -> float:
        """
        Combined gravitational attraction.
        High mass (noun/verb) + high resonance = strong pull.
        Low mass (article/filler) = ejected quickly.

        G_weight = σ(pos_mass × resonance_score)
        """
        res = self.resonance(token_vec)
        g = torch.sigmoid(
            torch.tensor(pos_mass * res, dtype=torch.float32)
        ).item()
        return g

    def half_life_decay(self, decay_rate: float = 0.01):
        """
        Apply exponential decay to Sun state.
        For conversation (fast decay): decay_rate=0.1
        For technical text (slow decay): decay_rate=0.01
        Call once per paragraph.
        """
        self.state = self.state * (1.0 - decay_rate)

    def __repr__(self):
        return (f"SunState(d={self.d}, alpha={self.alpha}, "
                f"age={self.age}, norm={self.state.norm().item():.4f})")
